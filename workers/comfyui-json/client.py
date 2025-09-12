import logging
import uuid
import random
from urllib.parse import urljoin
import json
import os
import posixpath as path

import requests
import argparse

from lib.test_utils import print_truncate_res
from utils.endpoint_util import Endpoint
from utils.ssl import get_cert_file_path
from .data_types import count_workload
from .prompt_config import styles, room_prompt, common_negative_prompts, style_id_mapping
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

def make_request(url: str, payload: dict, timeout: int = None, verify=True, context: str = "request"):
    """Helper function for making requests with consistent error handling"""
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=timeout,
            verify=verify
        )
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.HTTPError as http_err:
        log.error(f"HTTP error occurred during {context}: {http_err}")
        log.error(f"Status Code: {response.status_code}")
        log.error("Response content:", response.text)
        return None
    except requests.exceptions.Timeout:
        log.error(f"Timeout occurred during {context}: {url}")
        return None
    except requests.exceptions.ConnectionError:
        log.error(f"Connection error occurred during {context}: {url}")
        return None
    except json.JSONDecodeError as json_err:
        log.error(f"Failed to decode JSON response during {context}: {json_err}")
        if 'response' in locals():
            print("Response content:", response.text)
        return None
    except Exception as err:
        log.error(f"An unexpected error occurred during {context}: {err}")
        if 'response' in locals():
            log.error("Response content (if available):", response.text)
        return None



def build_prompt_str(prompt_list):
    return ", ".join(f"({text}:{weight})" for text, weight in prompt_list)

def resolve_style_name(style_input):
    """Convert numeric style_id to style name"""
    # If it's already a valid style name, return as is
    if style_input in styles:
        return style_input
    
    # If it's a number (string or int), map it to style name
    if style_input in style_id_mapping:
        return style_id_mapping[style_input]
    
    # If no mapping found, raise an error
    raise ValueError(f"Invalid style: {style_input}. Valid options: {list(styles.keys())} or numbers 0-2")

def call_custom_workflow_with_images(
    endpoint_group_name: str,
    api_key: str,
    server_url: str,
    workflow_path: str,
    user_img: str,
    style: str,
    room: str,
    prefix: str,
) -> dict:
    """Custom workflow with image uploads and prompt building"""

    WORKER_ENDPOINT = "/generate/sync"


    # This worker has concurrency = 1.  All workloads have cost value 1.0
    COST = count_workload()

    # Route to get worker URL
    route_payload = {
        "endpoint": endpoint_group_name,
        "api_key": api_key,
        "cost": COST,
    }
    
    # First request - get routing information
    route_response = make_request(
        url=urljoin(server_url, "/route/"),
        payload=route_payload,
        timeout=4,
        context="route request"
    )
    
    if route_response is None:
        return None
    
    if "url" not in route_response or not route_response["url"]:
        log.error("Error: No worker in 'Ready' state. Please wait while the serverless engine removes errored workers or finishes loading new workers.")
        return None
    
    if "status" in route_response:
        print(f"Autoscaler status: {route_response['status']}")
        return None
    
    # Extract data from route response
    url = route_response["url"]
    auth_data = dict(
        signature=route_response["signature"],
        cost=route_response["cost"],
        endpoint=route_response["endpoint"],
        reqnum=route_response["reqnum"],
        url=route_response["url"],
    )


    # Load workflow JSON
    with open(workflow_path, "r", encoding="utf-8") as f:
        prompt_json = json.load(f)


    # Upload images to server and get their filenames
    def upload_image(img_path):
        print(f"[UPLOAD] Starting upload for image: {img_path}")

        # Check if file exists and get file size
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[UPLOAD] Image file not found: {img_path}")

        file_size = os.path.getsize(img_path)
        print(f"[UPLOAD] File size: {file_size} bytes")

        upload_url = urljoin(url, "/upload/image")
        print(f"[UPLOAD] Upload URL: {upload_url}")

        try:
            with open(img_path, "rb") as f:
                files = {"file": (Path(img_path).name, f)}
                print(f"[UPLOAD] Sending POST request with file: {Path(img_path).name}")

                resp = requests.post(upload_url, files=files, verify=get_cert_file_path(), timeout=30)
                print(f"[UPLOAD] Response status code: {resp.status_code}")
                print(f"[UPLOAD] Response headers: {dict(resp.headers)}")

                if resp.status_code != 200:
                    print(f"[UPLOAD] Error response content: {resp.text}")
                    resp.raise_for_status()

                try:
                    response_json = resp.json()
                    print(f"[UPLOAD] Response JSON: {response_json}")

                    if "path" not in response_json:
                        raise ValueError(f"[UPLOAD] Server response missing 'path' field: {response_json}")

                    uploaded_path = response_json["path"]
                    print(f"[UPLOAD] Successfully uploaded image, server path: {uploaded_path}")
                    return uploaded_path

                except json.JSONDecodeError as e:
                    print(f"[UPLOAD] Failed to parse JSON response: {e}")
                    print(f"[UPLOAD] Raw response content: {resp.text}")
                    raise

        except requests.exceptions.RequestException as e:
            print(f"[UPLOAD] Network error during upload: {e}")
            raise
        except Exception as e:
            print(f"[UPLOAD] Unexpected error during upload: {e}")
            raise


    user_img_filename = upload_image(user_img)
    
    # Resolve style name from input (handle numeric style_id)
    resolved_style = resolve_style_name(style)
    style_img_path = path.join("/workspace/vast-pyworker", styles[resolved_style]["img"])
    style_img_filename = style_img_path

    prompt_json["14"]["inputs"]["image"] = user_img_filename
    prompt_json["31"]["inputs"]["image"] = style_img_filename

    # Build positive/negative prompt
    style_prompts = styles[resolved_style]["prompts"]
    room_pos = room_prompt[room]["positive"]
    room_neg = room_prompt[room]["negative"]
    positive_prompt = build_prompt_str(style_prompts + room_pos)
    negative_prompt = build_prompt_str(room_neg + common_negative_prompts)
    prompt_json["6"]["inputs"]["text"] = positive_prompt
    prompt_json["7"]["inputs"]["text"] = negative_prompt
    prompt_json["49"]["inputs"]["filename_prefix"] = prefix

    # Build the payload for the worker request
    taskId = str(uuid.uuid4())
    worker_payload = {
        "input": {
            "request_id": taskId,
            "workflow_json": prompt_json
        },
    }
    req_data = dict(payload=worker_payload, auth_data=auth_data)
    worker_url = urljoin(url, WORKER_ENDPOINT)
    print(f"url: {worker_url}")

    # Second request - call the worker endpoint
    worker_response = make_request(
        url=worker_url,
        payload=req_data,
        verify=get_cert_file_path(),
        context="worker request"
    )

    return worker_response

def call_text2image_workflow(
    endpoint_group_name: str, api_key: str, server_url: str
) -> None:
    """Simple Text2Image using the new modifier-based approach"""
    

    WORKER_ENDPOINT = "/generate/sync"

    # This worker has concurrency = 1.  All workloads have cost value 1.0
    COST = count_workload()
    
    # Route to get worker URL
    route_payload = {
        "endpoint": endpoint_group_name,
        "api_key": api_key,
        "cost": COST,
    }
    
    # First request - get routing information
    route_response = make_request(
        url=urljoin(server_url, "/route/"),
        payload=route_payload,
        timeout=4,
        context="route request"
    )
    
    if route_response is None:
        return None
    
    if "url" not in route_response or not route_response["url"]:
        log.error("Error: No worker in 'Ready' state. Please wait while the serverless engine removes errored workers or finishes loading new workers.")
        return None
    
    if "status" in route_response:
        print(f"Autoscaler status: {route_response['status']}")
        return None
    
    # Extract data from route response
    url = route_response["url"]
    auth_data = dict(
        signature=route_response["signature"],
        cost=route_response["cost"],
        endpoint=route_response["endpoint"],
        reqnum=route_response["reqnum"],
        url=route_response["url"],
    )
    
    # Build the payload for the worker request
    worker_payload = {
        "input": {
            "request_id": str(uuid.uuid4()),
            "modifier": "Text2Image",
            "modifications": {
                "prompt": "a beautiful landscape with mountains and lakes",
                "width": 1024,
                "height": 1024,
                "steps": 20,
                "seed": random.randint(0, 2**32 - 1)
            },
            "workflow_json": {}  # Empty since using modifier approach
        }
    }
    
    req_data = dict(payload=worker_payload, auth_data=auth_data)
    worker_url = urljoin(url, WORKER_ENDPOINT)
    print(f"url: {worker_url}")
    
    # Second request - call the worker endpoint
    worker_response = make_request(
        url=worker_url,
        payload=req_data,
        verify=get_cert_file_path(),
        context="worker request"
    )
    
    return worker_response


if __name__ == "__main__":
    from lib.test_utils import test_args

    parser = argparse.ArgumentParser(description="ComfyUI client with image upload and workflow override")
    parser.add_argument("-k", dest="api_key", type=str, required=True, help="Your vast account API key")
    parser.add_argument("-e", dest="endpoint_group_name", type=str, required=True, help="Endpoint group name")
    parser.add_argument("-l", dest="server_url", type=str, default="https://run.vast.ai", help="Server URL")
    parser.add_argument("-i", dest="instance", type=str, default="prod", help="Autoscaler shard, default: prod")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Submit workflow command
    submit_parser = subparsers.add_parser("submit", help="Submit a workflow")
    submit_parser.add_argument("--workflow", dest="workflow_path", type=str, required=True, help="Path to workflow JSON")
    submit_parser.add_argument("--user_img", dest="user_img", type=str, required=True, help="Path to user input image")
    submit_parser.add_argument("--style", dest="style", type=str, required=True, help="Style (style_eu1, style_jp1, style_country) or numeric ID (0, 1, 2)")
    submit_parser.add_argument("--room", dest="room", type=str, required=True, choices=list(room_prompt.keys()), help="Room type (living_room, dining_room, ...)")
    submit_parser.add_argument("--prefix", dest="prefix", type=str, required=True, help="Filename prefix for generated images")
    submit_parser.add_argument("--output", dest="output_file", type=str, required=True, help="Output file to save the submit result (JSON format)")
    
    # python3 -m workers.comfyui.client -k ... -e ... submit --workflow ... --user_img ... --style style_eu1 --room living_room --prefix ... --output submit_result.json

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        exit(1)
    
    endpoint_api_key = Endpoint.get_endpoint_api_key(
        endpoint_name=args.endpoint_group_name,
        account_api_key=args.api_key,
        instance=args.instance,
    )

    if endpoint_api_key:
        try:
            if args.command == "submit":
                res_json = call_custom_workflow_with_images(
                    endpoint_group_name=args.endpoint_group_name,
                    api_key=endpoint_api_key,
                    server_url=args.server_url,
                    workflow_path=args.workflow_path,
                    user_img=args.user_img,
                    style=args.style,
                    room=args.room,
                    prefix=args.prefix,
                )
                
                # Save result to output file
                with open(args.output_file, 'w', encoding="utf-8") as f:
                    json.dump(res_json, f, indent=2)
                
                print(f"Workflow submitted successfully! Result saved to: {args.output_file}")
                print(f"Task ID: {res_json.get('id', 'N/A')}")
                print(res_json)
        except Exception as e:
            log.error(f"Error during API call: {e}")
    else:
        log.error(f"Failed to get API key for endpoint {args.endpoint_group_name}")
