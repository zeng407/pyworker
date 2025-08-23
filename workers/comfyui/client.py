import logging
from urllib.parse import urljoin
import os
import base64
import time
import requests
from lib.test_utils import print_truncate_res
from utils.endpoint_util import Endpoint
from utils.ssl import get_cert_file_path

def save_images(res_json):
    if isinstance(res_json, dict) and "images" in res_json:
        os.makedirs("outputs", exist_ok=True)
        for idx, img_data in enumerate(res_json["images"]):
            if isinstance(img_data, str) and img_data.startswith("data:image/"):
                header, b64data = img_data.split(",", 1)
                ext = header.split("/")[1].split(";")[0]
            else:
                b64data = img_data
                ext = "png"
            out_path = os.path.join("outputs", f"output_{int(time.time())}_{idx}.{ext}")
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(b64data))
            print(f"Saved image to {out_path}")

"""
NOTE: this client example uses a custom comfy workflow compatible with SD3 only
"""
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


def call_default_workflow(
    endpoint_group_name: str, api_key: str, server_url: str
) -> None:
    WORKER_ENDPOINT = "/prompt"
    COST = 100
    route_payload = {
        "endpoint": endpoint_group_name,
        "api_key": api_key,
        "cost": COST,
    }
    response = requests.post(
        urljoin(server_url, "/route/"),
        json=route_payload,
        timeout=4,
    )
    response.raise_for_status()
    message = response.json()
    url = message["url"]
    auth_data = dict(
        signature=message["signature"],
        cost=message["cost"],
        endpoint=message["endpoint"],
        reqnum=message["reqnum"],
        url=message["url"],
    )
    payload = dict(
        prompt="a fat fluffy cat", width=1024, height=1024, steps=20, seed=123456789
    )
    req_data = dict(payload=payload, auth_data=auth_data)
    url = urljoin(url, WORKER_ENDPOINT)
    print(f"url: {url}")
    response = requests.post(
        url,
        json=req_data,
        verify=get_cert_file_path(),
    )
    response.raise_for_status()
    res_json = response.json()
    print_truncate_res(str(res_json))
    save_images(res_json)


def call_custom_workflow_for_sd3(
    endpoint_group_name: str, api_key: str, server_url: str
) -> None:
    WORKER_ENDPOINT = "/custom-workflow"
    COST = 100
    route_payload = {
        "endpoint": endpoint_group_name,
        "api_key": api_key,
        "cost": COST,
    }
    response = requests.post(
        urljoin(server_url, "/route/"),
        json=route_payload,
        timeout=4,
    )
    response.raise_for_status()
    message = response.json()
    url = message["url"]
    auth_data = dict(
        signature=message["signature"],
        cost=message["cost"],
        endpoint=message["endpoint"],
        reqnum=message["reqnum"],
        url=message["url"],
    )
    workflow = {
        "3": {
            "inputs": {
                "seed": 156680208700286,
                "steps": 20,
                "cfg": 8,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
            "class_type": "KSampler",
        },
        "4": {
            "inputs": {"ckpt_name": "sd3_medium_incl_clips_t5xxlfp16.safetensors"},
            "class_type": "CheckpointLoaderSimple",
        },
        "5": {
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
            "class_type": "EmptyLatentImage",
        },
        "6": {
            "inputs": {
                "text": "beautiful scenery nature glass bottle landscape, purple galaxy bottle",
                "clip": ["4", 1],
            },
            "class_type": "CLIPTextEncode",
        },
        "7": {
            "inputs": {"text": "text, watermark", "clip": ["4", 1]},
            "class_type": "CLIPTextEncode",
        },
        "8": {
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            "class_type": "VAEDecode",
        },
        "9": {
            "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
            "class_type": "SaveImage",
        },
    }
    # these values should match the values in the custom workflow above,
    # they are used to calculate workload
    custom_fields = dict(
        steps=20,
        width=512,
        height=512,
    )
    req_data = dict(
        payload=dict(custom_fields=custom_fields, workflow=workflow),
        auth_data=auth_data,
    )
    url = urljoin(url, WORKER_ENDPOINT)
    print(f"url: {url}")
    response = requests.post(
        url,
        json=req_data,
        verify=get_cert_file_path(),
    )
    response.raise_for_status()
    res_json = response.json()
    print_truncate_res(str(res_json))
    save_images(res_json)


if __name__ == "__main__":
    from lib.test_utils import test_args

    args = test_args.parse_args()
    endpoint_api_key = Endpoint.get_endpoint_api_key(
        endpoint_name=args.endpoint_group_name,
        account_api_key=args.api_key,
        instance=args.instance,
    )
    if endpoint_api_key:
        try:
            call_default_workflow(
                api_key=endpoint_api_key,
                endpoint_group_name=args.endpoint_group_name,
                server_url=args.server_url,
            )
            call_custom_workflow_for_sd3(
                api_key=endpoint_api_key,
                endpoint_group_name=args.endpoint_group_name,
                server_url=args.server_url,
            )
        except Exception as e:
            log.error(f"Error during API call: {e}")
    else:
        log.error(f"Failed to get API key for endpoint {args.endpoint_group_name} ")
