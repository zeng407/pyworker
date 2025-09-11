import os
import logging
import dataclasses
import base64
from typing import Optional, Union, Type

from aiohttp import web, ClientResponse

from lib.backend import Backend, LogAction
from lib.data_types import EndpointHandler
from lib.server import start_server
from .data_types import ComfyWorkflowData
from .data_types import ImageUploadData


MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://0.0.0.0:38188")

# This is the last log line that gets emitted once comfyui+extensions have been fully loaded
MODEL_SERVER_START_LOG_MSG = "To see the GUI go to: "
MODEL_SERVER_ERROR_LOG_MSGS = [
    "MetadataIncompleteBuffer",  # This error is emitted when the downloaded model is corrupted
    "Value not in list: ",  # This error is emitted when the model file is not there at all
]


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


async def generate_client_response(
        client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        # Check if the response is actually streaming based on response headers/content-type
        is_streaming_response = (
            model_response.content_type == "text/event-stream"
            or model_response.content_type == "application/x-ndjson"
            or model_response.headers.get("Transfer-Encoding") == "chunked"
            or "stream" in model_response.content_type.lower()
        )

        if is_streaming_response:
            log.debug("Detected streaming response...")
            res = web.StreamResponse()
            res.content_type = model_response.content_type
            await res.prepare(client_request)
            async for chunk in model_response.content:
                await res.write(chunk)
            await res.write_eof()
            log.debug("Done streaming response")
            return res
        else:
            log.debug("Detected non-streaming response...")
            content = await model_response.read()
            return web.Response(
                body=content,
                status=model_response.status,
                content_type=model_response.content_type
            )
            

@dataclasses.dataclass
class ComfyWorkflowHandler(EndpointHandler[ComfyWorkflowData]):

    @property
    def endpoint(self) -> str:
        return "/generate/sync"

    @property
    def healthcheck_endpoint(self) -> Optional[str]:
        return "/health"

    @classmethod
    def payload_cls(cls) -> Type[ComfyWorkflowData]:
        return ComfyWorkflowData

    def make_benchmark_payload(self) -> ComfyWorkflowData:
        return ComfyWorkflowData.for_test()

    async def generate_client_response(
        self, client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        return await generate_client_response(client_request, model_response)

class ImageUploadHandler:
    """
    Custom handler for image upload that processes multipart form data
    and saves uploaded images to the uploads directory.
    
    This handler follows the backend metrics pattern for workload tracking.
    """
    
    backend: 'Backend' = None  # Will be injected when creating the handler
    
    @property
    def endpoint(self) -> str:
        return ""  # No backend endpoint needed for file upload
    
    async def handle_request(self, request):
        """
        Process multipart image upload request with metrics tracking
        """
        # Create a dummy auth_data since image upload doesn't require authentication
        # but we need it for metrics tracking
        from lib.data_types import AuthData
        import asyncio
        
        auth_data = AuthData(
            signature="", cost="0", endpoint=self.endpoint, 
            reqnum=1, url=""
        )
        
        reader = await request.multipart()
        
        filename = None
        file_data = None
        
        # Process all fields in the multipart request
        async for field in reader:
            if field.name == "file":
                filename = field.filename
                if not filename:
                    log.debug("No filename provided")
                    return web.json_response({"error": "No filename provided"}, status=400)
                # Read all data from the file field
                file_data = await field.read()
            elif field.name == "name":
                # Optional custom name
                custom_name = (await field.text()).strip()
                if custom_name:
                    filename = custom_name
        
        # Check if we received file data
        if not file_data:
            log.debug("No file data received")
            return web.json_response({"error": "No file data received"}, status=400)
        
        if not filename:
            log.debug("No filename provided")
            return web.json_response({"error": "No filename provided"}, status=400)
        
        # Create ImageUploadData for logging/metrics
        upload_data = ImageUploadData(filename=filename, file_data=file_data)
        workload = upload_data.count_workload()
        
        # Handle client disconnection
        async def cancel_if_disconnected():
            await request.wait_for_disconnection()
            log.debug(f"upload request with reqnum: {auth_data.reqnum} was canceled")
            self.backend.metrics._request_canceled(workload=workload)
            return web.Response(status=500)
        
        async def process_upload():
            # Start metrics tracking
            self.backend.metrics._request_start(workload=workload, reqnum=auth_data.reqnum)
            
            try:
                # Save to uploads directory
                uploads_dir = os.path.abspath("uploads")
                os.makedirs(uploads_dir, exist_ok=True)
                save_path = os.path.join(uploads_dir, filename)
                
                # Write file data
                with open(save_path, "wb") as f:
                    f.write(file_data)
                
                # Verify file was written correctly
                if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
                    log.debug(f"Failed to save file: {save_path}")
                    self.backend.metrics._request_errored(workload=workload)
                    return web.json_response({"error": "Failed to save file"}, status=500)
                
                log.debug(f"Successfully uploaded file: {save_path} (size: {os.path.getsize(save_path)} bytes)")
                self.backend.metrics._request_success(workload=workload)
                return web.json_response({"success": True, "filename": filename, "path": save_path})
                
            except Exception as e:
                log.debug(f"Error during upload: {e}")
                self.backend.metrics._request_errored(workload=workload)
                return web.json_response({"error": str(e)}, status=500)
            finally:
                # End metrics tracking
                self.backend.metrics._request_end(workload=workload, reqnum=auth_data.reqnum)
        
        # Race between upload processing and client disconnection
        done, pending = await asyncio.wait(
            [cancel_if_disconnected(), process_upload()],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel any remaining tasks
        for task in pending:
            task.cancel()
            
        # Return the result from the completed task
        return await done.pop()


backend = Backend(
    model_server_url=MODEL_SERVER_URL,
    model_log_file=os.environ["MODEL_LOG"],
    allow_parallel_requests=False,
    benchmark_handler=ComfyWorkflowHandler(
        benchmark_runs=3, benchmark_words=100
    ),
    log_actions=[
        (LogAction.ModelLoaded, MODEL_SERVER_START_LOG_MSG),
        (LogAction.Info, "Downloading:"),
        *[
            (LogAction.ModelError, error_msg)
            for error_msg in MODEL_SERVER_ERROR_LOG_MSGS
        ],
    ],
)


async def handle_ping(_):
    return web.Response(body="pong")


routes = [
    web.post("/generate/sync", backend.create_handler(ComfyWorkflowHandler())),
    web.get("/ping", handle_ping),
    web.post("/upload/image", ImageUploadHandler(backend=backend).handle_request),
]

if __name__ == "__main__":
    start_server(backend, routes)
