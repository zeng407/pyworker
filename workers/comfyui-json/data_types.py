import os
import sys
import random
import dataclasses
from typing import Dict, Any
from functools import cache
from math import ceil

from lib.data_types import ApiPayload, JsonDataException
import json


@dataclasses.dataclass
class ImageUploadData(ApiPayload):
    filename: str
    file_data: bytes = dataclasses.field(repr=False)  # Don't show file data in repr
    
    @classmethod
    def for_test(cls) -> "ImageUploadData":
        return cls(filename="test_image.png", file_data=b"fake_image_data")
    
    def get_cost(self) -> int:
        return 0  # Image upload should be free
    
    def get_word_count(self) -> int:
        return 1  # Minimal word count for upload operations
    
    def count_workload(self) -> float:
        # Image upload is a lightweight operation, use minimal workload
        # Approximately equivalent to a very small image generation request
        return 10.0
    
    def generate_payload_json(self) -> Dict[str, Any]:
        # Image upload doesn't need to send payload to backend since endpoint is ""
        # Return minimal JSON for compatibility
        return {
            "filename": self.filename,
            "file_size": len(self.file_data)
        }
    
    
    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "ImageUploadData":
        # This won't be used for multipart uploads, but required by base class
        errors = {}
        if "filename" not in json_msg:
            errors["filename"] = "missing parameter"
        if errors:
            raise JsonDataException(errors)
        return cls(filename=json_msg["filename"], file_data=b"")


with open("workers/comfyui/misc/test_prompts.txt", "r") as f:
    test_prompts = f.readlines()

def count_workload() -> float:
    # Always 100.0 where there is a single instance of ComfyUI handling requests
    # Results will indicate % or a job completed per second.  Avoids sub 0.1 sec performance indication
    return 100.0



BENCHMARK_WORKFLOW_PATH = os.path.join(
    os.path.dirname(__file__), "misc", "interior_design_v0.03_linux_benchmark.json"
)

def load_benchmark_workflow():
    with open(BENCHMARK_WORKFLOW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@dataclasses.dataclass
class ComfyWorkflowData(ApiPayload):
    input: dict

    @classmethod
    def for_test(cls):
        """
        Use the variables available to simulate workflows of the required running time
        Example: SD1.5, simple image gen 10000 steps, 512px x 512px will run for approximately 9 minutes @ ~18 it/s (RTX 4090)
        """
        workflow_json = load_benchmark_workflow()
        return cls(
            input={
                "request_id": f"test-{random.randint(1000, 99999)}",
                "workflow_json": workflow_json
            }
        )

    def generate_payload_json(self) -> Dict[str, Any]:
        # input is already a dict, just return it wrapped in the expected structure
        return {"input": self.input}

    def count_workload(self) -> float:
        return count_workload()

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "ComfyWorkflowData":
        # Extract required fields
        if "input" not in json_msg:
            raise JsonDataException({"input": "missing parameter"})
        
        return cls(
            input=json_msg["input"]
        )