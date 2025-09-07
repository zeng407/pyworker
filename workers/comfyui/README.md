This is the base PyWorker for comfyui. It can be used to create PyWorker that use various models and
workflows. It provides two endpoints:

1. `/prompt`: Uses the default comfy workflow defined under `misc/default_workflows`
2. `/custom_workflow`: Allows the client to send their own comfy workflow with each API request.

To use the comfyui PyWorker, `$COMFY_MODEL` env variable must be set in the template. Current options are
`sd3` and `flux`. Each have example clients.

To add new models, a JSON with name `$COMFY_MODEL.json` must be created under `misc/default_workflows`

NOTE: default workflows follow this format:

```json
{
  "input": {
    "handler": "RawWorkflow",
    "aws_access_key_id": "your-s3-access-key",
    "aws_secret_access_key": "your-s3-secret-access-key",
    "aws_endpoint_url": "https://my-endpoint.backblaze.com",
    "aws_bucket_name": "your-bucket",
    "webhook_url": "your-webhook-url",
    "webhook_extra_params": {},
    "workflow_json": {}
  }
}
```

You can ignore all of these fields except for `workflow_json`.

Fields written as "{{FOO}}" will be replaced using data from a user request. For example, SD3's workflow has the
following nodes:

```json
      "5": {
        "inputs": {
          "width": "{{WIDTH}}",
          "height": "{{HEIGHT}}",
          "batch_size": 1
        },

      "6": {
        "inputs": {
          "text": "{{PROMPT}}",
          "clip": ["11", 0]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
          "title": "CLIP Text Encode (Prompt)"
        }
      },
      ...
      "17": {
        "inputs": {
          "scheduler": "simple",
          "steps": "{{STEPS}}",
          "denoise": 1,
          "model": ["12", 0]
        },
        "class_type": "BasicScheduler",
        "_meta": {
          "title": "BasicScheduler"
        }
      },
      ...
      "25": {
        "inputs": {
          "noise_seed": "{{SEED}}"
        },
        "class_type": "RandomNoise",
        "_meta": {
          "title": "RandomNoise"
        }
      }

```

Incoming requests have the following JSON format:

```json
{
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
}
```

Each value in those fields with replace the placeholder of the same name in the default workflow.

## Usage Examples

The ComfyUI worker provides a command-line client with multiple operations:

### 1. Submit Workflow

Submit an interior design workflow with a custom image (results saved to JSON file):

```bash
python3 -m workers.comfyui.client -k "<YOUR_API_KEY>" -e "comfyui-test" submit \
  --workflow "workers/comfyui/misc/interior_design_v0.03_linux.json" \
  --user_img "tests/room1.jpeg" \
  --style style_eu1 \
  --room living_room \
  --prefix PREFIX \
  --output submit_result.json
```

**Style Options:**
- **Named styles**: `style_eu1`, `style_jp1`, `style_country`
- **Numeric IDs**: `0` (style_eu1), `1` (style_jp1), `2` (style_country)

This will return a task ID that you can use to query status and download results.

### 2. Query Task Status

Check the status of a submitted task (results saved to JSON file):

```bash
python3 -m workers.comfyui.client -k "<YOUR_API_KEY>" -e "comfyui-test" query \
  --task_id 99533104-3947-47b6-8f2c-d41a35b5ed75 \
  --output query_result.json
```

The response will include:
- Task status (`completed`, `processing`, `failed`, etc.)
- Output file paths when completed
- Any error messages

### 3. Download Generated Images

Download specific images from completed tasks:

```bash
# Download a specific image
python3 -m workers.comfyui.client -k "<YOUR_API_KEY>" -e "comfyui-test" download \
  --path "99533104-3947-47b6-8f2c-d41a35b5ed75/TASK_ID_1_00004_.png"

# Download all images from a task automatically
python3 -m workers.comfyui.client -k "<YOUR_API_KEY>" -e "comfyui-test" download-all \
  --task_id 99533104-3947-47b6-8f2c-d41a35b5ed75 \
  --output "my_generated_images"
```

### 4. Pipeline Operations

Chain commands using JSON output for automation:

```bash
# Query and pipe to download command
python3 -m workers.comfyui.client -k "<YOUR_API_KEY>" -e "comfyui-test" query \
  --task_id 99533104-3947-47b6-8f2c-d41a35b5ed75 \
  --output query_result.json \
  --json | python3 -m workers.comfyui.client -k "<YOUR_API_KEY>" -e "comfyui-test" download-from-json
```

### Complete Workflow Example

```bash
# 1. Submit workflow and save task info
python3 -m workers.comfyui.client -k "your-api-key" -e "comfyui-test" submit \
  --workflow "workers/comfyui/misc/interior_design_v0.03_linux.json" \
  --user_img "input.jpg" \
  --style 0 \
  --room living_room \
  --prefix "LIVING_ROOM" \
  --output task_info.json

# 2. Extract task ID and query status
TASK_ID=$(jq -r '.id' task_info.json)
python3 -m workers.comfyui.client -k "your-api-key" -e "comfyui-test" query \
  --task_id "$TASK_ID" \
  --output status.json

# 3. Download all generated images
python3 -m workers.comfyui.client -k "your-api-key" -e "comfyui-test" download-all \
  --task_id "$TASK_ID" \
  --output "results"
```

**Security Note:** The download endpoint includes path traversal protection to ensure files can only be accessed from the ComfyUI output directory (`/opt/ComfyUI/output/`).

### Available Commands

- **`submit`**: Submit a new workflow for processing (requires `--output` to save results)
- **`query`**: Query the status of an existing task by task ID (requires `--output` to save results)  
- **`download`**: Download a specific image file by path
- **`download-all`**: Query task status and download all output files automatically
- **`download-from-json`**: Download files from JSON input (useful for piping operations)

See Vast's serverless documentation for more details on how to use comfyui with autoscaler
