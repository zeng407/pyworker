from aiohttp import web
import os
import logging
import dataclasses
import base64
from typing import Optional, Union, Type

from aiohttp import web, ClientResponse
from anyio import open_file

from lib.backend import Backend, LogAction
from lib.data_types import EndpointHandler
from lib.server import start_server
from .data_types import DefaultComfyWorkflowData, CustomComfyWorkflowData
import time
import re

# /upload/image endpoint for uploading images to be used as input
async def handle_upload_image(request):
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
        return web.json_response({"error": "Failed to save file"}, status=500)
    
    log.debug(f"Successfully uploaded file: {save_path} (size: {os.path.getsize(save_path)} bytes)")
    return web.json_response({"success": True, "filename": filename, "path": save_path})


MODEL_SERVER_URL = "http://0.0.0.0:38188"

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
            if "output" not in res:
                return web.json_response(
                    data=dict(error="there was an error in the workflow"),
                    status=422,
                )
            image_paths = [path["local_path"] for path in res["output"]["images"]]
            if not image_paths:
                return web.json_response(
                    data=dict(error="workflow did not produce any images"),
                    status=422,
                )
            images = []
            for image_path in image_paths:
                async with await open_file(image_path, mode="rb") as f:
                    contents = await f.read()
                    images.append(
                        f"data:image/png;base64,{base64.b64encode(contents).decode('utf-8')}"
                    )
            return web.json_response(data=dict(images=images))
        case code:
            log.debug("SENDING RESPONSE: ERROR: unknown code")
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
        return "/runsync"

    @property
    def healthcheck_endpoint(self) -> Optional[str]:
        return None

    @classmethod
    def payload_cls(cls) -> Type[CustomComfyWorkflowData]:
        return CustomComfyWorkflowData

    def make_benchmark_payload(self) -> CustomComfyWorkflowData:
        return CustomComfyWorkflowData.for_test()

    def _convert_image_paths_to_absolute(self, workflow: dict) -> dict:
        """Convert relative image paths in workflow to absolute paths"""
        import copy
        workflow_copy = copy.deepcopy(workflow)
        
        for node_id, node_data in workflow_copy.items():
            if isinstance(node_data, dict) and "inputs" in node_data:
                inputs = node_data["inputs"]
                if isinstance(inputs, dict) and "image" in inputs:
                    image_path = inputs["image"]
                    if isinstance(image_path, str) and not os.path.isabs(image_path):
                        # Convert relative path to absolute path
                        abs_path = os.path.abspath(image_path)
                        inputs["image"] = abs_path
                        log.debug(f"Converted image path: {image_path} -> {abs_path}")
        
        return workflow_copy

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


async def handle_ping(_):
    return web.Response(body="pong")


routes = [
    web.post("/prompt", backend.create_handler(DefaultComfyWorkflowHandler())),
    web.post("/custom-workflow", backend.create_handler(CustomComfyWorkflowHandler())),
    web.get("/ping", handle_ping),
    web.post("/upload/image", handle_upload_image),
]

if __name__ == "__main__":
    start_server(backend, routes)
