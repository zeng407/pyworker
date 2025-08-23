from aiohttp import web
# /upload/image endpoint for uploading images to be used as input
async def handle_upload_image(request):
    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != "file":
        return web.json_response({"error": "Missing file field"}, status=400)
    filename = field.filename
    # Get the custom name from the form (optional)
    name_field = await reader.next()
    if name_field and name_field.name == "name":
        custom_name = (await name_field.text()).strip()
        if custom_name:
            filename = custom_name
    # Save to uploads directory
    os.makedirs("uploads", exist_ok=True)
    save_path = os.path.join("uploads", filename)
    with open(save_path, "wb") as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            f.write(chunk)
    return web.json_response({"success": True, "filename": filename, "path": save_path})
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
def decode_and_save_base64_images(workflow_json):
    """
    Find all LoadImage nodes with base64 image strings, decode and save to disk, replace with filename.
    """
    img_pattern = re.compile(r"^data:image/(png|jpeg|jpg);base64,(.*)$", re.IGNORECASE)
    for node_id, node in workflow_json.items():
        if node.get("class_type") == "LoadImage":
            img_val = node.get("inputs", {}).get("image")
            if isinstance(img_val, str):
                m = img_pattern.match(img_val)
                if m:
                    ext = m.group(1)
                    b64data = m.group(2)
                    filename = f"uploaded_{node_id}_{int(time.time())}.{ext}"
                    filepath = os.path.join("uploads", filename)
                    os.makedirs("uploads", exist_ok=True)
                    with open(filepath, "wb") as f:
                        f.write(base64.b64decode(b64data))
                    node["inputs"]["image"] = filename
    return workflow_json


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

    async def generate_client_response(
        self, client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        # Intercept and decode base64 images in workflow JSON before passing to backend
        try:
            data = await client_request.json()
            if "input" in data and "workflow_json" in data["input"]:
                wf = data["input"]["workflow_json"]
                if "workflow" in wf:
                    wf["workflow"] = decode_and_save_base64_images(wf["workflow"])
        except Exception as e:
            log.error(f"Error decoding base64 images: {e}")
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
