# ComfyUI PyWorker

This is the base PyWorker for ComfyUI. It provides a unified interface for running any ComfyUI workflow through a proxy-based architecture.

## Endpoint

The worker provides a single endpoint:

- `/generate/sync`: Processes ComfyUI workflows using either predefined modifiers or custom workflow JSON

## Request Format

The worker accepts requests in the following format. Choose either modifier mode OR custom workflow mode:

**Modifier Mode:**
```json
{
  "input": {
    "request_id": "uuid-string",    // optional - UUID generated if not provided
    "modifier": "RawWorkflow",
    "modifications": {
      "prompt": "a beautiful landscape",
      "width": 1024,
      "height": 1024,
      "steps": 20,
      "seed": 123456789
    },
    "s3": { ... },       // optional
    "webhook": { ... }   // optional
  },
  "expected_time": 30.0
}
```

**Custom Workflow Mode:**
```json
{
  "input": {
    "request_id": "uuid-string",    // optional - UUID generated if not provided
    "workflow_json": {
      // Complete ComfyUI workflow JSON
    },
    "s3": { ... },       // optional
    "webhook": { ... }   // optional
  },
  "expected_time": 30.0
}
```

## Request Fields

### Required Fields

- **`input`**: Contains the main workflow data
- **`input.request_id`**: Unique identifier for the request
- **`expected_time`**: Expected runtime in seconds on RTX4090 (defaults to 46.0 if not provided)

### Workflow Mode (Choose One)

You must provide either `modifier` OR `workflow_json`, but not both:

#### Option 1: Modifier Mode
- **`input.modifier`**: Name of the predefined workflow modifier (e.g., "Text2Image")
- **`input.modifications`**: Parameters to pass to the modifier

#### Option 2: Custom Workflow Mode  
- **`input.workflow_json`**: Complete ComfyUI workflow JSON

### Optional Fields

- **`input.s3`**: S3 configuration for file storage
- **`input.webhook`**: Webhook configuration for notifications

These configurations can be provided in the request JSON or via environment variables. Request-level configuration takes precedence over environment variables.

#### S3 Configuration

**Via Request JSON:**
```json
"s3": {
  "access_key_id": "your-s3-access-key",
  "secret_access_key": "your-s3-secret-access-key", 
  "endpoint_url": "https://my-endpoint.backblaze.com",
  "bucket_name": "your-bucket",
  "region": "us-east-1"
}
```

**Via Environment Variables:**
```bash
S3_ACCESS_KEY_ID=your-key
S3_SECRET_ACCESS_KEY=your-secret
S3_BUCKET_NAME=your-bucket
S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_REGION=us-east-1
```

#### Webhook Configuration

**Via Request JSON:**
```json
"webhook": {
  "url": "your-webhook-url",
  "extra_params": {
    "custom_field": "value"
  }
}
```

**Via Environment Variables:**
```bash
WEBHOOK_URL=https://your-webhook.com  # Default webhook URL
WEBHOOK_TIMEOUT=30                   # Webhook timeout in seconds
```

## Examples

### Basic Text-to-Image (Modifier Mode)

```json
{
  "input": {
    "modifier": "Text2Image",
    "modifications": {
      "prompt": "a cat sitting on a windowsill",
      "width": 512,
      "height": 512,
      "steps": 20,
      "seed": 42
    }
  },
  "expected_time": 25.0
}
```

### Custom Workflow Mode

```json
{
  "input": {
    "request_id": "67890",    // optional - using custom ID for tracking
    "workflow_json": {
      "3": {
        "inputs": {
          "seed": 42,
          "steps": 20,
          "cfg": 8,
          "sampler_name": "euler",
          "scheduler": "normal",
          "denoise": 1,
          "model": ["4", 0],
          "positive": ["6", 0],
          "negative": ["7", 0],
          "latent_image": ["5", 0]
        },
        "class_type": "KSampler"
      }
    }
  },
  "expected_time": 45.0
}
```

## Expected Time Guidelines

The `expected_time` field helps with resource planning and should reflect expected runtime on RTX4090:

- **Simple text-to-image**: 15-30 seconds
- **Complex workflows with upscaling**: 60+ seconds  
- **Video generation**: 180+ seconds
- **Default**: 46 seconds (if not specified)

## Client Libraries

See the test client examples for implementation details on how to integrate with the ComfyUI worker.

---

See Vast's serverless documentation for more details on how to use ComfyUI with autoscaler.