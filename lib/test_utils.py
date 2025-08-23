import logging
import os
import time
import argparse
from typing import Callable, List, Dict, Tuple, Dict, Any, Type
from time import sleep
import threading
from enum import Enum
from collections import Counter
from dataclasses import dataclass, field, asdict
from urllib.parse import urljoin
from utils.endpoint_util import Endpoint
from utils.ssl import get_cert_file_path
import requests

from lib.data_types import AuthData, ApiPayload

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


class ClientStatus(Enum):
    FetchEndpoint = 1
    Generating = 2
    Done = 3
    Error = 4


total_success = 0
last_res = []
stop_event = threading.Event()

start_time = time.time()
test_args = argparse.ArgumentParser(description="Test inference endpoint")
test_args.add_argument(
    "-k", dest="api_key", type=str, required=True, help="Your vast account API key"
)
test_args.add_argument(
    "-e",
    dest="endpoint_group_name",
    type=str,
    required=True,
    help="Endpoint group name",
)
test_args.add_argument(
    "-l",
    dest="server_url",
    action="store_const",
    const="http://localhost:8081",
    default="https://run.vast.ai",
    help="Call local autoscaler instead of prod, for dev use only",
)
test_args.add_argument(
    "-i",
    dest="instance",
    type=str,
    default="prod",
    help="Autoscaler shard to run the command against, default: prod",
)

GetPayloadAndWorkload = Callable[[], Tuple[Dict[str, Any], float]]


def print_truncate_res(res: str):
    if len(res) > 150:
        print(f"{res[:50]}....{res[-100:]}")
    else:
        print(res)


@dataclass
class ClientState:
    endpoint_group_name: str
    api_key: str
    server_url: str
    worker_endpoint: str
    instance: str
    payload: ApiPayload
    url: str = ""
    status: ClientStatus = ClientStatus.FetchEndpoint
    as_error: List[str] = field(default_factory=list)
    infer_error: List[str] = field(default_factory=list)
    conn_errors: Counter = field(default_factory=Counter)

    def save_images(self, res_json: Dict[str, Any]):
        if isinstance(res_json, dict) and "images" in res_json:
            import os, base64
            os.makedirs("outputs", exist_ok=True)
            for idx, img_data in enumerate(res_json["images"]):
                if img_data.startswith("data:image/"):
                    header, b64data = img_data.split(",", 1)
                    ext = header.split("/")[1].split(";")[0]
                else:
                    b64data = img_data
                    ext = "png"
                out_path = os.path.join("outputs", f"output_{int(time.time())}_{idx}.{ext}")
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(b64data))
                print(f"Saved image to {out_path}")

    def make_call(self):
        self.status = ClientStatus.FetchEndpoint
        if not self.api_key:
            self.as_error.append(
                f"Endpoint {self.endpoint_group_name} not found for API key",
            )
            self.status = ClientStatus.Error
            return
        route_payload = {
            "endpoint": self.endpoint_group_name,
            "api_key": self.api_key,
            "cost": self.payload.count_workload(),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            urljoin(self.server_url, "/route/"),
            json=route_payload,
            headers=headers,
            timeout=4,
        )
        if response.status_code != 200:
            self.as_error.append(
                f"code: {response.status_code}, body: {response.text}",
            )
            self.status = ClientStatus.Error
            return
        message = response.json()
        worker_address = message["url"]
        req_data = dict(
            payload=asdict(self.payload),
            auth_data=asdict(AuthData.from_json_msg(message)),
        )
        self.url = worker_address
        url = urljoin(worker_address, self.worker_endpoint)
        self.status = ClientStatus.Generating

        response = requests.post(
            url,
            json=req_data,
            verify=get_cert_file_path(),
        )
        if response.status_code != 200:
            self.infer_error.append(
                f"code: {response.status_code}, body: {response.text}, url: {url}",
            )
            self.status = ClientStatus.Error
            return
        res_json = response.json()
        res = str(res_json)

        self.save_images(res_json)
        global total_success
        global last_res
        total_success += 1
        last_res.append(res)
        self.status = ClientStatus.Done

    def simulate_user(self) -> None:
        try:
            self.make_call()
        except Exception as e:
            print(e)
            self.status = ClientStatus.Error
            _ = e
            self.conn_errors[self.url] += 1


def print_state(clients: List[ClientState], num_clients: int) -> None:
    print("starting up...")
    sleep(2)
    center_size = 14
    global start_time
    while len(clients) < num_clients or (
        any(
            map(
                lambda client: client.status
                in [ClientStatus.FetchEndpoint, ClientStatus.Generating],
                clients,
            )
        )
    ):
        sleep(0.5)
        os.system("clear")
        print(
            " | ".join(
                [member.name.center(center_size) for member in ClientStatus]
                + [
                    item.center(center_size)
                    for item in [
                        "urls",
                        "as_error",
                        "infer_error",
                        "conn_error",
                        "total_success",
                    ]
                ]
            )
        )
        unique_urls = len(set([c.url for c in clients if c.url != ""]))
        as_errors = sum(
            map(
                lambda client: len(client.as_error),
                [client for client in clients],
            )
        )
        infer_errors = sum(
            map(
                lambda client: len(client.infer_error),
                [client for client in clients],
            )
        )
        conn_errors = sum([client.conn_errors for client in clients], start=Counter())
        conn_errors_str = ",".join(map(str, conn_errors.values())) or "0"
        elapsed = time.time() - start_time
        print(
            " | ".join(
                map(
                    lambda item: str(item).center(center_size),
                    [
                        len(list(filter(lambda x: x.status == member, clients)))
                        for member in ClientStatus
                    ]
                    + [
                        unique_urls,
                        as_errors,
                        infer_errors,
                        conn_errors_str,
                        f"{total_success}({((total_success/elapsed) * 60):.2f}/minute)",
                    ],
                )
            )
        )
        if conn_errors:
            print("conn_errors:")
            for url, count in conn_errors.items():
                print(url.ljust(28), ": ", str(count))
        elapsed = time.time() - start_time
        print(f"\n elapsed: {int(elapsed // 60)}:{int(elapsed % 60)}")
        if last_res:
            for i, res in enumerate(last_res[-10:]):
                print_truncate_res(f"res #{1+i+max(len(last_res )-10,0)}: {res}")
        if stop_event.is_set():
            print("\n### waiting for existing connections to close ###")


def run_test(
    num_requests: int,
    requests_per_second: int,
    endpoint_group_name: str,
    api_key: str,
    server_url: str,
    worker_endpoint: str,
    payload_cls: Type[ApiPayload],
    instance: str,
):
    threads = []

    clients = []
    print_thread = threading.Thread(target=print_state, args=(clients, num_requests))
    print_thread.daemon = True  # makes threads get killed on program exit
    print_thread.start()
    endpoint_api_key = Endpoint.get_endpoint_api_key(
        endpoint_name=endpoint_group_name, account_api_key=api_key, instance=instance
    )
    if not endpoint_api_key:
        log.debug(f"Endpoint {endpoint_group_name} not found for API key")
        return
    try:
        for _ in range(num_requests):
            client = ClientState(
                endpoint_group_name=endpoint_group_name,
                api_key=endpoint_api_key,
                server_url=server_url,
                worker_endpoint=worker_endpoint,
                payload=payload_cls.for_test(),
                instance=instance,
            )
            clients.append(client)
            thread = threading.Thread(target=client.simulate_user, args=())
            threads.append(thread)
            thread.start()
            sleep(1 / requests_per_second)
        for thread in threads:
            thread.join()
        print("done spawning workers")
    except KeyboardInterrupt:
        stop_event.set()


def test_load_cmd(
    payload_cls: Type[ApiPayload], endpoint: str, arg_parser: argparse.ArgumentParser
):
    arg_parser.add_argument(
        "-n",
        dest="num_requests",
        type=int,
        required=True,
        help="total number of requests",
    )
    arg_parser.add_argument(
        "-rps",
        dest="requests_per_second",
        type=float,
        required=True,
        help="requests per second",
    )
    args = arg_parser.parse_args()
    if hasattr(args, "comfy_model"):
        os.environ["COMFY_MODEL"] = args.comfy_model
    server_url = dict(
        prod="https://run.vast.ai",
        alpha="https://run-alpha.vast.ai",
        candidate="https://run-candidate.vast.ai",
        local="http://localhost:8080",
    )[args.instance]
    run_test(
        num_requests=args.num_requests,
        requests_per_second=args.requests_per_second,
        api_key=args.api_key,
        server_url=server_url,
        endpoint_group_name=args.endpoint_group_name,
        worker_endpoint=endpoint,
        payload_cls=payload_cls,
        instance=args.instance,
    )
