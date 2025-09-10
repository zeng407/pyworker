import aiohttp
import os
import logging
import dataclasses
import base64
from typing import Optional, Union, Type

from aiohttp import web, ClientResponse

from lib.backend import Backend, LogAction
from lib.data_types import EndpointHandler
from lib.server import start_server
from .data_types import DefaultComfyWorkflowData, CustomComfyWorkflowData
import time
import re
import os
import time
from typing import Optional, Type, Union

from .data_types import ImageUploadData


@dataclasses.dataclass  
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
            signature="", cost="100", endpoint=self.endpoint, 
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

MODEL_SERVER_URL = "http://127.0.0.1:18288" # API Wrapper Service

# This is the last log line that gets emitted once comfyui+extensions have been fully loaded
MODEL_SERVER_START_LOG_MSG = "To see the GUI go to: http://127.0.0.1:18188"
MODEL_SERVER_ERROR_LOG_MSGS = [
    "MetadataIncompleteBuffer",  # This error is emitted when the downloaded model is corrupted
    "Value not in list: unet_name",  # This error is emitted when the model file is not there at all
]


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


async def generate_client_response(
    request: web.Request, response: ClientResponse
) -> Union[web.Response, web.StreamResponse]:
    _ = request
    match response.status:
        case 200:
            log.debug("SUCCESS")
            res = await response.json()
            log.debug(f"Model server response JSON: {res}")
            
            # Return only essential fields for task status
            simplified_response = {
                "id": res.get("id"),
                "message": res.get("message"),
                "status": res.get("status"),
                "output": res.get("output", [])
            }
            return web.json_response(simplified_response)
        case 202:
            # Accepted, but not completed. Return the content as JSON.
            res = await response.json()
            log.debug(f"SENDING RESPONSE: 202 Accepted, content: {res}")
            return web.json_response(res, status=202)
        case code:
            log.debug(f"SENDING RESPONSE: ERROR: unknown code {code}")
            return web.Response(status=code)


@dataclasses.dataclass
class DefaultComfyWorkflowHandler(EndpointHandler[DefaultComfyWorkflowData]):

    @property
    def endpoint(self) -> str:
        return "/runsync"

    @property
    def healthcheck_endpoint(self) -> Optional[str]:
        return None

    @classmethod
    def payload_cls(cls) -> Type[DefaultComfyWorkflowData]:
        return DefaultComfyWorkflowData

    def make_benchmark_payload(self) -> DefaultComfyWorkflowData:
        return DefaultComfyWorkflowData.for_test()

    async def generate_client_response(
        self, client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        return await generate_client_response(client_request, model_response)


@dataclasses.dataclass
class CustomComfyWorkflowHandler(EndpointHandler[CustomComfyWorkflowData]):

    @property
    def endpoint(self) -> str:
        return "/payload"

    @property
    def healthcheck_endpoint(self) -> Optional[str]:
        return None

    @classmethod
    def payload_cls(cls) -> Type[CustomComfyWorkflowData]:
        return CustomComfyWorkflowData

    def make_benchmark_payload(self) -> CustomComfyWorkflowData:
        return CustomComfyWorkflowData.for_test()

    async def generate_client_response(
        self, client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        return await generate_client_response(client_request, model_response)


backend = Backend(
    model_server_url=MODEL_SERVER_URL,
    model_log_file=os.environ["MODEL_LOG"],
    allow_parallel_requests=False,
    benchmark_handler=DefaultComfyWorkflowHandler(
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

async def handle_download_output(request):
    # Get the path from the URL, e.g., /99533104-3947-47b6-8f2c-d41a35b5ed75/TASK_ID_1_00004_.png
    path = request.match_info.get('path')
    if not path:
        return web.json_response({'error': 'No path provided'}, status=400)
    
    # Security: Prevent path traversal attacks
    # Normalize the path and ensure it doesn't contain '..' or absolute paths
    normalized_path = os.path.normpath(path)
    if normalized_path.startswith('/') or '..' in normalized_path or normalized_path.startswith('..'):
        log.error(f"Invalid path detected (potential path traversal): {path}")
        return web.json_response({'error': 'Invalid path'}, status=400)
    
    # Full file path in ComfyUI output directory
    base_dir = os.path.abspath("/opt/ComfyUI/output")
    full_path = os.path.join(base_dir, normalized_path)
    
    # Ensure the resolved path is still within the base directory
    if not full_path.startswith(base_dir):
        log.error(f"Path traversal attempt detected: {path} -> {full_path}")
        return web.json_response({'error': 'Access denied'}, status=403)
    
    log.debug(f"Attempting to download file: {full_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(full_path):
            log.error(f"File not found: {full_path}")
            return web.json_response({'error': f'File not found: {path}'}, status=404)
        
        # Check if it's a file (not directory)
        if not os.path.isfile(full_path):
            log.error(f"Path is not a file: {full_path}")
            return web.json_response({'error': f'Path is not a file: {path}'}, status=400)
        
        # Read and return the file
        with open(full_path, 'rb') as f:
            content = f.read()
        
        # Determine content type based on file extension
        filename = os.path.basename(normalized_path)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            content_type = 'image/png' if filename.lower().endswith('.png') else 'image/jpeg'
        else:
            content_type = 'application/octet-stream'
        
        log.debug(f"Successfully serving file: {full_path} (size: {len(content)} bytes)")
        return web.Response(
            body=content,
            content_type=content_type,
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )
        
    except Exception as e:
        log.error(f"Error downloading file {full_path}: {e}")
        return web.json_response({'error': str(e)}, status=500)

routes = [
    web.post("/prompt", backend.create_handler(DefaultComfyWorkflowHandler())),
    web.post("/custom-workflow", backend.create_handler(CustomComfyWorkflowHandler())),
    web.post("/upload/image", ImageUploadHandler(backend=backend).handle_request),
]

if __name__ == "__main__":
    start_server(backend, routes)
