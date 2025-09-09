import sys
import os
import json
import random
import dataclasses
import inspect
from typing import Dict, Any
from functools import cache
from math import ceil
from enum import Enum

from lib.data_types import ApiPayload, JsonDataException


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
        return 1.0
    
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


@dataclasses.dataclass
class TaskInfoData(ApiPayload):
    task_id: str
    
    @classmethod
    def for_test(cls) -> "TaskInfoData":
        return cls(task_id="test-task-id-12345")
    
    def get_cost(self) -> int:
        return 0  # Task info query should be free
    
    def get_word_count(self) -> int:
        return 1  # Minimal word count for query operations
    
    def count_workload(self) -> float:
        # Task info query is a very lightweight operation
        return 0.1
    
    def generate_payload_json(self) -> Dict[str, Any]:
        # For task info, we don't send JSON payload, task_id is in URL path
        return {"task_id": self.task_id}
    
    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "TaskInfoData":
        errors = {}
        if "task_id" not in json_msg:
            errors["task_id"] = "missing parameter"
        if errors:
            raise JsonDataException(errors)
        return cls(task_id=json_msg["task_id"])


with open("workers/comfyui/misc/test_prompts.txt", "r", encoding="utf-8") as f:
    test_prompts = f.readlines()


class Model(Enum):
    Flux = "flux"
    Sd3 = "sd3"

    def get_request_time(self) -> int:
        match self:
            case Model.Flux:
                return 23
            case Model.Sd3:
                return 6


@cache
def get_model() -> Model:
    match os.environ.get("COMFY_MODEL"):
        case "flux":
            return Model.Flux
        case "sd3":
            return Model.Sd3
        case None:
            raise Exception(
                "For comfyui pyworker, $COMFY_MODEL must be set in the vast template"
            )
        case model:
            raise Exception(f"Unsupported comfyui model: {model}")


@cache
def get_request_template() -> str:
    with open(f"workers/comfyui/misc/default_workflows/{get_model().value}.json", encoding="utf-8") as f:
        return f.read()


def count_workload(width: int, height: int, steps: int) -> float:
    """
    we want to normalize the workload is a number such that cur_perf(tokens/second) for 1024x1024 image with
    28 steps is 200 tokens on a 4090.

    in order get that we calculate the

    A = ( absolute workload based on given data )
    B = ( absolute workload for a 1024x1024 image with 28 steps )

    and adjust the workload to 200 tokens by A/B.

    we then adjust for difference between Flux and SD3 by multiplying this value by expected request time for a
    standard image(23s for Flux, 6s for SD3).
    On a 4090, this would give us a workload that would give a cur_perf(workload / request_time) of around 200
    """

    def _calculate_absolute_tokens(width_: int, height_: int, steps_: int) -> float:
        """
        This is based on how openai counts image generation tokens, see: https://openai.com/api/pricing/

        we count how many 512x512 grids are needed to cover the image.
        each tile is then counted as 175 tokens.
        each image generation also has constant of 85 base tokens.

        we then adjust the count based on the number of steps. The baseline number of steps is assumed to be 28.
        Some testing with flux gave me this data:

        steps(X)  | request time(Y)
        __________|_________________
        07(0.25x) | 11s (0.47x)
        14(0.50x) | 15s (0.65x)
        21(0.75x) | 20s (0.86x)
        28(1.00x) | 23s (1.00x)
        35(1.25x) | 28s (1.21x)
        42(1.50x) | 32s (1.39x)
        49(1.75x) | 37s (1.60x)

        this gives a linear regression of Y = 0.61*X + 6.57

        we can use this as an adjustment_factor for token count

        adjustment_factor = (0.61 * steps + 6.57)
        """

        width_grids = ceil(width_ / 512)
        height_grids = ceil(height_ / 512)
        tokens = 85 + width_grids * height_grids * 175
        adjustment_factor = 0.61 * steps_ + 6.57
        return tokens * adjustment_factor

    REQUEST_TIME_FOR_STANDARD_IMAGE = get_model().get_request_time()

    absolute_tokens = _calculate_absolute_tokens(
        width_=width, height_=height, steps_=steps
    )
    absolute_tokens_standard_image = _calculate_absolute_tokens(
        width_=1024, height_=1024, steps_=28
    )
    return REQUEST_TIME_FOR_STANDARD_IMAGE * (
        (absolute_tokens / absolute_tokens_standard_image) * 200
    )


@dataclasses.dataclass
class DefaultComfyWorkflowData(ApiPayload):
    prompt: str
    width: int
    height: int
    steps: int
    seed: int

    @classmethod
    def for_test(cls):

        test_prompt = random.choice(test_prompts).rstrip()
        return cls(
            prompt=test_prompt,
            width=1024,
            height=1024,
            steps=28,
            seed=random.randint(0, sys.maxsize),
        )

    def generate_payload_json(
        self,
    ) -> Dict[str, Any]:
        return json.loads(
            get_request_template()
            .replace("{{PROMPT}}", self.prompt)
            # these values should be of int type. Since "{{VAR}}" is wrapped with " in the template
            # to make the JSON valid, we must replace the double quotes. i.e. "{{WIDTH}}" -> 1024 and not "1024"
            .replace('"{{WIDTH}}"', str(self.width))
            .replace('"{{HEIGHT}}"', str(self.height))
            .replace('"{{STEPS}}"', str(self.steps))
            .replace('"{{SEED}}"', str(self.seed))
        )

    def count_workload(self) -> float:
        return count_workload(width=self.width, height=self.height, steps=self.steps)

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "DefaultComfyWorkflowData":
        errors = {}
        for param in inspect.signature(cls).parameters:
            if param not in json_msg:
                errors[param] = "missing parameter"
        if errors:
            raise JsonDataException(errors)
        return cls(
            **{
                k: v
                for k, v in json_msg.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclasses.dataclass
class CustomComfyWorkflowData(ApiPayload):
    custom_fields: Dict[str, int]
    workflow: Dict[str, Any]

    @classmethod
    def for_test(cls):
        raise NotImplementedError("Custom comfy workflow is not used for testing")

    def count_workload(self) -> float:
        return count_workload(
            width=int(self.custom_fields.get("width", 1024)),
            height=int(self.custom_fields.get("height", 1024)),
            steps=int(self.custom_fields.get("steps", 28)),
        )

    def generate_payload_json(self) -> Dict[str, Any]:
        template_json = json.loads(get_request_template())
        # Convert relative image paths to absolute paths in workflow
        workflow_with_abs_paths = self._convert_image_paths_to_absolute(self.workflow)
        template_json["input"]["workflow_json"] = workflow_with_abs_paths
        return template_json

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
        
        return workflow_copy

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "CustomComfyWorkflowData":
        errors = {}
        for param in inspect.signature(cls).parameters:
            if param not in json_msg:
                errors[param] = "missing parameter"
        if errors:
            raise JsonDataException(errors)
        return cls(
            **{
                k: v
                for k, v in json_msg.items()
                if k in inspect.signature(cls).parameters
            }
        )
