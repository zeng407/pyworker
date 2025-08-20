import sys
import json
import random
import dataclasses
import inspect
from typing import Dict, Any
from functools import cache
from math import ceil

from lib.data_types import ApiPayload, JsonDataException


with open("workers/comfyui/misc/test_prompts.txt", "r") as f:
    test_prompts = f.readlines()

@dataclasses.dataclass
class ComfyWorkflowData(ApiPayload):
    input: dict
    expected_time: float = 46.0  # Default: 2x baseline (23s * 2) for RTX4090

    @classmethod
    def for_test(cls):
        test_prompt = random.choice(test_prompts).rstrip()
        return cls(
            input={
                "request_id": f"test-{random.randint(1000, 9999)}",
                "modifier": "Text2Image",
                "modifications": {
                    "prompt": test_prompt,
                    "width": 1024,
                    "height": 1024,
                    "steps": 28,
                    "seed": random.randint(0, sys.maxsize),
                }
            },
            expected_time=25.0  # Test data: expect 25 seconds on RTX4090 (slightly above baseline)
        )

    def generate_payload_json(self) -> Dict[str, Any]:
        # input is already a dict, just return it wrapped in the expected structure
        return {"input": self.input}

    def count_workload(self) -> float:
        """
        This needs review. We cannot reasonably predict the workload based on the inputs. We may be processing:
        - Images
        - Videos
        - Audio... There may also be complex loops in the workflow.

        User will provide an expected time to complete and we will calculate equivalent cost

        Convert user-provided expected_time (RTX4090 seconds) to the old scoring system.
        
        The old system normalized to: 1024x1024, 28 steps = 200 tokens on RTX4090
        The old formula was: REQUEST_TIME_FOR_STANDARD_IMAGE * (time_ratio * 200)
        
        Now the user provides the expected request time directly.
        Default expected_time is 46s (2x baseline) if not specified.
        """
        # Baseline: standard image (1024x1024, 28 steps) = 23s = 200 tokens on RTX4090
        RTX4090_BASELINE_TIME = 23.0  # seconds for standard image on RTX4090
        BASELINE_TOKENS = 200  # tokens for standard image
        
        # Calculate time ratio compared to baseline
        time_ratio = self.expected_time / RTX4090_BASELINE_TIME
        
        # Return workload score: time_ratio * baseline tokens
        return time_ratio * BASELINE_TOKENS

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "ComfyWorkflowData":
        # Extract required fields
        if "input" not in json_msg:
            raise JsonDataException({"input": "missing parameter"})
        
        # expected_time is optional, uses default if not provided
        expected_time = json_msg.get("expected_time", 46.0)  # Default: 2x baseline
        
        return cls(
            input=json_msg["input"],
            expected_time=float(expected_time)
        )