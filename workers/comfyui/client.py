import logging
from urllib.parse import urljoin
import os
import base64
import time
import json
import argparse
from workers.comfyui.prompt_config import styles, room_prompt, common_negative_prompts
from pathlib import Path
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



def build_prompt_str(prompt_list):
    return ", ".join(f"({text}:{weight})" for text, weight in prompt_list)

def call_custom_workflow_with_images(
    endpoint_group_name: str,
    api_key: str,
    server_url: str,
    workflow_path: str,
    user_img: str,
    style: str,
    room: str,
    task_id: str,
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

    # Load workflow JSON
    with open(workflow_path, "r") as f:
        prompt_json = json.load(f)



    # Upload images to server and get their filenames
    def upload_image(img_path):
        files = {"file": open(img_path, "rb")}
        data = {"name": Path(img_path).name}
        upload_url = urljoin(server_url, "/upload/image")
        resp = requests.post(upload_url, files=files, data=data)
        resp.raise_for_status()
        return resp.json()["filename"]

    user_img_filename = upload_image(user_img)
    style_img_path = styles[style]["img"]
    style_img_filename = upload_image(style_img_path)

    prompt_json["14"]["inputs"]["image"] = user_img_filename
    prompt_json["31"]["inputs"]["image"] = style_img_filename

    # Build positive/negative prompt
    style_prompts = styles[style]["prompts"]
    room_pos = room_prompt[room]["positive"]
    room_neg = room_prompt[room]["negative"]
    positive_prompt = build_prompt_str(style_prompts + room_pos)
    negative_prompt = build_prompt_str(room_neg + common_negative_prompts)
    prompt_json["6"]["inputs"]["text"] = positive_prompt
    prompt_json["7"]["inputs"]["text"] = negative_prompt
    prompt_json["49"]["inputs"]["filename_prefix"] = task_id

    # You may want to update custom_fields based on workflow or user input
    custom_fields = dict(
        steps=prompt_json["3"]["inputs"].get("steps", 20),
        width=1024,
        height=1024,
    )

    req_data = dict(
        payload=dict(custom_fields=custom_fields, workflow=prompt_json),
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
    parser = argparse.ArgumentParser(description="ComfyUI client with image upload and workflow override")
    parser.add_argument("-k", dest="api_key", type=str, required=True, help="Your vast account API key")
    parser.add_argument("-e", dest="endpoint_group_name", type=str, required=True, help="Endpoint group name")
    parser.add_argument("-l", dest="server_url", type=str, default="https://run.vast.ai", help="Server URL")
    parser.add_argument("-i", dest="instance", type=str, default="prod", help="Autoscaler shard, default: prod")
    parser.add_argument("--workflow", dest="workflow_path", type=str, required=True, help="Path to workflow JSON")
    parser.add_argument("--user_img", dest="user_img", type=str, required=True, help="Path to user input image")
    parser.add_argument("--style", dest="style", type=str, required=True, choices=list(styles.keys()), help="Style (style_eu1, style_jp1, style_country)")
    parser.add_argument("--room", dest="room", type=str, required=True, choices=list(room_prompt.keys()), help="Room type (living_room, dining_room, ...)")
    parser.add_argument("--task_id", dest="task_id", type=str, required=True, help="Task ID for filename prefix")

    # python3 -m workers.comfyui.client -k ... -e ... --workflow ... --user_img ... --style style_eu1 --room living_room --task_id ...
    args = parser.parse_args()
    endpoint_api_key = Endpoint.get_endpoint_api_key(
        endpoint_name=args.endpoint_group_name,
        account_api_key=args.api_key,
        instance=args.instance,
    )
    if endpoint_api_key:
        try:
            call_custom_workflow_with_images(
                endpoint_group_name=args.endpoint_group_name,
                api_key=endpoint_api_key,
                server_url=args.server_url,
                workflow_path=args.workflow_path,
                user_img=args.user_img,
                style=args.style,
                room=args.room,
                task_id=args.task_id,
            )
        except Exception as e:
            log.error(f"Error during API call: {e}")
    else:
        log.error(f"Failed to get API key for endpoint {args.endpoint_group_name} ")
