import logging
import uuid
import random
from urllib.parse import urljoin

import requests

from lib.test_utils import print_truncate_res
from utils.endpoint_util import Endpoint
from utils.ssl import get_cert_file_path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


def call_text2image_workflow(
    endpoint_group_name: str, api_key: str, server_url: str
) -> None:
    """Simple Text2Image using the new modifier-based approach"""
    WORKER_ENDPOINT = "/generate/sync"
    COST = 100
    
    # Route to get worker URL
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
    
    # Build the new payload structure
    payload = {
        "input": {
            "request_id": str(uuid.uuid4()),
            "modifier": "RawWorkflow",  # or whatever your Text2Image modifier is called
            "modifications": {
                "prompt": "a beautiful landscape with mountains and lakes",
                "width": 1024,
                "height": 1024,
                "steps": 20,
                "seed": random.randint(0, 2**32 - 1)
            },
            "workflow_json": {}  # Empty since using modifier approach
        },
        "expected_time": 30.0  # Expected 30 seconds on RTX4090
    }
    
    req_data = dict(payload=payload, auth_data=auth_data)
    url = urljoin(url, WORKER_ENDPOINT)
    print(f"url: {url}")
    
    response = requests.post(
        url,
        json=req_data,
        verify=get_cert_file_path(),
    )
    response.raise_for_status()
    print_truncate_res(str(response.json()))


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
            call_text2image_workflow(
                api_key=endpoint_api_key,
                endpoint_group_name=args.endpoint_group_name,
                server_url=args.server_url,
            )
        except Exception as e:
            log.error(f"Error during API call: {e}")
    else:
        log.error(f"Failed to get API key for endpoint {args.endpoint_group_name}")