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

def query_task_status(
    endpoint_group_name: str,
    api_key: str,
    server_url: str,
    task_id: str,
) -> dict:
    """Query task status by task ID"""
    COST = 0  # Query operations should be free
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
    
    # Query task status
    task_url = urljoin(url, f"/task/{task_id}")
    print(f"Querying task status: {task_url}")
    response = requests.get(
        task_url,
        verify=get_cert_file_path(),
    )
    response.raise_for_status()
    return response.json()


def download_file(
    endpoint_group_name: str,
    api_key: str,
    server_url: str,
    file_path: str,
    output_dir: str = "downloads"
) -> None:
    """Download a file from ComfyUI output directory"""
    route_payload = {
        "endpoint": endpoint_group_name,
        "api_key": api_key,
        "cost": 0,  # Download should be free
    }
    response = requests.post(
        urljoin(server_url, "/route/"),
        json=route_payload,
        timeout=4,
    )
    response.raise_for_status()
    message = response.json()
    url = message["url"]
    
    # Download the file
    download_url = urljoin(url, f"/download/{file_path}")
    print(f"Downloading from: {download_url}")
    
    response = requests.get(
        download_url,
        verify=get_cert_file_path(),
        stream=True
    )
    response.raise_for_status()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract filename from path
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, filename)
    
    # Save the file
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    file_size = os.path.getsize(output_path)
    print(f"Successfully downloaded: {output_path} (size: {file_size} bytes)")


def download_all_task_files(
    endpoint_group_name: str,
    api_key: str,
    server_url: str,
    task_id: str,
    output_dir: str = "downloads"
) -> None:
    """Query task status and download all output files"""
    print(f"Querying task {task_id} and downloading all files...")
    
    # First, query the task status
    task_status = query_task_status(
        endpoint_group_name=endpoint_group_name,
        api_key=api_key,
        server_url=server_url,
        task_id=task_id,
    )
    
    print("Task status:")
    print(task_status)
    
    # Extract file paths from output
    if "output" not in task_status or not task_status["output"]:
        print("No output files found in task status")
        return
    
    downloaded_files = []
    for output_item in task_status["output"]:
        file_path = None
        
        if isinstance(output_item, dict):
            if "local_path" in output_item:
                # Extract relative path from local_path
                # e.g., "/opt/ComfyUI/output/99533104-3947-47b6-8f2c-d41a35b5ed75/TASK_ID_1_00001_.png"
                # becomes "99533104-3947-47b6-8f2c-d41a35b5ed75/TASK_ID_1_00001_.png"
                local_path = output_item["local_path"]
                if local_path.startswith("/opt/ComfyUI/output/"):
                    file_path = local_path[len("/opt/ComfyUI/output/"):]
                else:
                    # Fallback: just use the filename
                    file_path = os.path.basename(local_path)
            elif "filename" in output_item:
                # Handle the old format with filename and subfolder
                if "subfolder" in output_item and output_item["subfolder"]:
                    file_path = f"{output_item['subfolder']}/{output_item['filename']}"
                else:
                    file_path = output_item["filename"]
        
        if file_path:
            print(f"Downloading: {file_path}")
            try:
                download_file(
                    endpoint_group_name=endpoint_group_name,
                    api_key=api_key,
                    server_url=server_url,
                    file_path=file_path,
                    output_dir=output_dir,
                )
                downloaded_files.append(file_path)
            except Exception as e:
                print(f"Error downloading {file_path}: {e}")
        else:
            print(f"Could not extract file path from output item: {output_item}")
    
    print(f"Downloaded {len(downloaded_files)} files to {output_dir}/")
    for file_path in downloaded_files:
        print(f"  - {file_path}")


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
        upload_url = urljoin(url, "/upload/image")
        with open(img_path, "rb") as f:
            files = {"file": (Path(img_path).name, f)}
            resp = requests.post(upload_url, files=files, verify=get_cert_file_path())
            resp.raise_for_status()
            return resp.json()["path"]

    user_img_filename = upload_image(user_img)
    style_img_path = styles[style]["img"]
    style_img_filename = style_img_path

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
    prompt_json["49"]["inputs"]["filename_prefix"] = prefix

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
    print(f"auth_data: {auth_data}")
    response = requests.post(
        url,
        json=req_data,
        verify=get_cert_file_path(),
    )
    response.raise_for_status()
    res_json = response.json()
    # print_truncate_res(str(res_json))
    # save_images(res_json)
    return res_json



if __name__ == "__main__":
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
    submit_parser.add_argument("--style", dest="style", type=str, required=True, choices=list(styles.keys()), help="Style (style_eu1, style_jp1, style_country)")
    submit_parser.add_argument("--room", dest="room", type=str, required=True, choices=list(room_prompt.keys()), help="Room type (living_room, dining_room, ...)")
    submit_parser.add_argument("--prefix", dest="prefix", type=str, required=True, help="Filename prefix for generated images")
    
    # Query task status command
    query_parser = subparsers.add_parser("query", help="Query task status")
    query_parser.add_argument("--task_id", dest="task_id", type=str, required=True, help="Task ID to query")
    query_parser.add_argument("--json", dest="output_json", action="store_true", help="Output file paths as JSON for piping to other commands")
    
    # Download file command
    download_parser = subparsers.add_parser("download", help="Download a file from ComfyUI output")
    download_parser.add_argument("--path", dest="file_path", type=str, required=True, help="File path relative to ComfyUI output directory (e.g., 99533104-3947-47b6-8f2c-d41a35b5ed75/TASK_ID_1_00004_.png)")
    download_parser.add_argument("--output", dest="output_dir", type=str, default="downloads", help="Output directory for downloaded files (default: downloads)")
    
    # Download all files from a task command
    download_all_parser = subparsers.add_parser("download-all", help="Query task status and download all output files")
    download_all_parser.add_argument("--task_id", dest="task_id", type=str, required=True, help="Task ID to query and download files from")
    download_all_parser.add_argument("--output", dest="output_dir", type=str, default="downloads", help="Output directory for downloaded files (default: downloads)")
    
    # Download from JSON input (for piping)
    download_json_parser = subparsers.add_parser("download-from-json", help="Download files from JSON input (for piping)")
    download_json_parser.add_argument("--json", dest="json_input", type=str, help="JSON string with file paths (if not provided, reads from stdin)")
    download_json_parser.add_argument("--output", dest="output_dir", type=str, default="downloads", help="Output directory for downloaded files (default: downloads)")

    # python3 -m workers.comfyui.client -k ... -e ... submit --workflow ... --user_img ... --style style_eu1 --room living_room --prefix ...
    # python3 -m workers.comfyui.client -k ... -e ... query --task_id ...
    # python3 -m workers.comfyui.client -k ... -e ... download --path 99533104-3947-47b6-8f2c-d41a35b5ed75/TASK_ID_1_00004_.png
    # python3 -m workers.comfyui.client -k ... -e ... download-all --task_id 99533104-3947-47b6-8f2c-d41a35b5ed75
    # 
    # Pipeline examples:
    # python3 -m workers.comfyui.client -k ... -e ... query --task_id 99533104-3947-47b6-8f2c-d41a35b5ed75 --json | python3 -m workers.comfyui.client -k ... -e ... download-from-json
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
                print("Workflow submitted successfully:")
                print(res_json)
            elif args.command == "query":
                res_json = query_task_status(
                    endpoint_group_name=args.endpoint_group_name,
                    api_key=endpoint_api_key,
                    server_url=args.server_url,
                    task_id=args.task_id,
                )
                
                if args.output_json:
                    # Extract file paths for piping
                    file_paths = []
                    if "output" in res_json and res_json["output"]:
                        for output_item in res_json["output"]:
                            if isinstance(output_item, dict):
                                if "local_path" in output_item:
                                    # Extract relative path from local_path
                                    local_path = output_item["local_path"]
                                    if local_path.startswith("/opt/ComfyUI/output/"):
                                        file_path = local_path[len("/opt/ComfyUI/output/"):]
                                    else:
                                        file_path = os.path.basename(local_path)
                                    file_paths.append(file_path)
                                elif "filename" in output_item:
                                    # Handle the old format
                                    if "subfolder" in output_item and output_item["subfolder"]:
                                        file_path = f"{output_item['subfolder']}/{output_item['filename']}"
                                    else:
                                        file_path = output_item["filename"]
                                    file_paths.append(file_path)
                    
                    import json
                    print(json.dumps({
                        "task_id": args.task_id,
                        "status": res_json.get("status"),
                        "file_paths": file_paths
                    }))
                else:
                    print("Task status:")
                    print(res_json)
            elif args.command == "download":
                download_file(
                    endpoint_group_name=args.endpoint_group_name,
                    api_key=endpoint_api_key,
                    server_url=args.server_url,
                    file_path=args.file_path,
                    output_dir=args.output_dir,
                )
                print("File downloaded successfully!")
            elif args.command == "download-all":
                download_all_task_files(
                    endpoint_group_name=args.endpoint_group_name,
                    api_key=endpoint_api_key,
                    server_url=args.server_url,
                    task_id=args.task_id,
                    output_dir=args.output_dir,
                )
                print("All files downloaded successfully!")
            elif args.command == "download-from-json":
                import json
                import sys
                
                # Get JSON input
                if args.json_input:
                    json_data = json.loads(args.json_input)
                else:
                    # Read from stdin
                    json_data = json.loads(sys.stdin.read())
                
                file_paths = json_data.get("file_paths", [])
                print(f"Downloading {len(file_paths)} files from JSON input...")
                
                downloaded_files = []
                for file_path in file_paths:
                    print(f"Downloading: {file_path}")
                    try:
                        download_file(
                            endpoint_group_name=args.endpoint_group_name,
                            api_key=endpoint_api_key,
                            server_url=args.server_url,
                            file_path=file_path,
                            output_dir=args.output_dir,
                        )
                        downloaded_files.append(file_path)
                    except Exception as e:
                        print(f"Error downloading {file_path}: {e}")
                
                print(f"Downloaded {len(downloaded_files)} files successfully!")
        except Exception as e:
            log.error(f"Error during API call: {e}")
    else:
        log.error(f"Failed to get API key for endpoint {args.endpoint_group_name} ")
