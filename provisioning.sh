#!/bin/bash

# This file will be sourced in init.sh

# https://raw.githubusercontent.com/ai-dock/comfyui/main/config/provisioning/sd3.sh

# Packages are installed after nodes so we can fix them...

if [ -z "${HF_TOKEN}" ]; then
    echo "HF_TOKEN is not set. Exiting."
    exit 1
fi

PYTHON_PACKAGES=(
    "comfyui-frontend-package==1.23.4"
    "comfyui-workflow-templates==0.1.41"
    "comfyui-embedded-docs==0.2.4"
    "torch"
    "torchsde"
    "torchvision"
    "torchaudio"
    "numpy<2"
    "einops"
    "transformers>=4.37.2"
    "tokenizers>=0.13.3"
    "sentencepiece"
    "safetensors>=0.4.2"
    "aiohttp>=3.11.8"
    "yarl>=1.18.0"
    "pyyaml"
    "Pillow"
    "scipy"
    "tqdm"
    "psutil"
    "alembic"
    "SQLAlchemy"
    # Non essential dependencies
    "kornia>=0.7.1"
    "spandrel"
    "soundfile"
    "av>=14.2.0"
    "pydantic~=2.0"
    "pydantic-settings~=2.0"
    "diffusers"
    "opencv-python<4.10.0"
)

NODES=(
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/cubiq/ComfyUI_IPAdapter_plus"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/yolain/ComfyUI-Easy-Use"
    "https://github.com/chrisgoringe/cg-use-everywhere"
    "https://github.com/neverbiasu/ComfyUI-SAM2"
    "https://github.com/cubiq/ComfyUI_essentials"
)

CHECKPOINT_MODELS=(
    "https://huggingface.co/a34384300/XSarchitectural-InteriorDesign-ForXSLora/resolve/main/xsarchitectural_v11.ckpt"
    # "https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp16.safetensors"
    # "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"
    # "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors"
    # "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors"
    # "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors"
    # "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)
CHECKPOINT_MODELS_SDXL=(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"
)

LORA_MODELS=(
    #"https://civitai.com/api/download/models/16576"
    "https://civitai.com/api/download/models/30384" #xsarchitectural-7.safetensors

)

VAE_MODELS_SDXL=(
    # "https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-560000-ema-pruned.safetensors"
    # "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"
    "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"
)

ESRGAN_MODELS=(
    # "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth"
    # "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth"
    # "https://huggingface.co/Akumetsu971/SD_Anime_Futuristic_Armor/resolve/main/4x_NMKD-Siax_200k.pth"
)

CONTROLNET_MODELS=(
    # "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_canny-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_depth-fp16.safetensors"
    # "https://huggingface.co/kohya-ss/ControlNet-diff-modules/resolve/main/diff_control_sd15_depth_fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_hed-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_mlsd-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_normal-fp16.safetensors"
    # "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_openpose-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_scribble-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_seg-fp16.safetensors"
    # "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_canny-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_color-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_depth-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_keypose-fp16.safetensors"
    # "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_openpose-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_seg-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_sketch-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_style-fp16.safetensors"
   
)

CONTROLNET_MODELS_15=(
    "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors"
    "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors"
   
)
CONTROLNET_MODELS_SDXL_CANNY=(
    "https://huggingface.co/xinsir/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model_V2.safetensors"
)
CONTROLNET_MODELS_SDXL_DEPTH=(
    "https://huggingface.co/xinsir/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors"
)

CLIP_VERSION_MODELS=(
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors|CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" # CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
    # "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors"
)

SAMS_MODELS=(
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)

IPADAPTER_MODELS=(
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors"
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors"
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors"
)

### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###

function provisioning_start() {
    DISK_GB_AVAILABLE=$(($(df --output=avail -m "${WORKSPACE}" | tail -n1) / 1000))
    DISK_GB_USED=$(($(df --output=used -m "${WORKSPACE}" | tail -n1) / 1000))
    DISK_GB_ALLOCATED=$(($DISK_GB_AVAILABLE + $DISK_GB_USED))
    provisioning_print_header
    provisioning_get_nodes
    provisioning_install_python_packages
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/ckpt" \
        "${CHECKPOINT_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/ckpt/SDXL" \
        "${CHECKPOINT_MODELS_SDXL[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/loras" \
        "${LORA_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/controlnet" \
        "${CONTROLNET_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/controlnet/1.5" \
        "${CONTROLNET_MODELS_15[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/controlnet/SDXL/controlnet-canny-sdxl-1.0" \
        "${CONTROLNET_MODELS_SDXL_CANNY[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/controlnet/SDXL/controlnet-depth-sdxl-1.0" \
        "${CONTROLNET_MODELS_SDXL_DEPTH[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/vae/SDXL" \
        "${VAE_MODELS_SDXL[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/esrgan" \
        "${ESRGAN_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/clip_vision" \
        "${CLIP_VERSION_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/sams" \
        "${SAMS_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/storage/stable_diffusion/models/ipadapter" \
        "${IPADAPTER_MODELS[@]}"
    provisioning_print_end
}

function provisioning_get_nodes() {
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="/opt/ComfyUI/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                printf "Updating node: %s...\n" "${repo}"
                ( cd "$path" && git pull )
                if [[ -e $requirements ]]; then
                    "$COMFYUI_VENV_PIP" install -r "$requirements"
                fi
            fi
        else
            printf "Downloading node: %s...\n" "${repo}"
            git clone "${repo}" "${path}" --recursive
            if [[ -e $requirements ]]; then
                "$COMFYUI_VENV_PIP" install -r "${requirements}"
            fi
        fi
    done
}

function provisioning_install_python_packages() {
    if [ ${#PYTHON_PACKAGES[@]} -gt 0 ]; then
        "$COMFYUI_VENV_PIP" install --no-cache-dir \
            ${PYTHON_PACKAGES[*]}
    fi
}

function provisioning_get_models() {
    if [[ -z $2 ]]; then return 1; fi
    dir="$1"
    mkdir -p "$dir"
    shift
    if [[ $DISK_GB_ALLOCATED -ge $DISK_GB_REQUIRED ]]; then
        arr=("$@")
    else
        printf "WARNING: Low disk space allocation - Only the first model will be downloaded!\n"
        arr=("$1")
    fi

    printf "Downloading %s model(s) to %s...\n" "${#arr[@]}" "$dir"
    for url in "${arr[@]}"; do
        # Support custom filename: "url|filename"
        if [[ "$url" == *"|"* ]]; then
            model_url="${url%%|*}"
            model_filename="${url##*|}"
        else
            model_url="$url"
            model_filename=""
        fi
        printf "Downloading: %s\n" "${model_url}"
        provisioning_download "${model_url}" "${dir}" "${model_filename}"
        printf "\n"
    done
}

function provisioning_print_header() {
    printf "\n##############################################\n#                                            #\n#          Provisioning container            #\n#                                            #\n#         This will take some time           #\n#                                            #\n# Your container will be ready on completion #\n#                                            #\n##############################################\n\n"
    if [[ $DISK_GB_ALLOCATED -lt $DISK_GB_REQUIRED ]]; then
        printf "WARNING: Your allocated disk size (%sGB) is below the recommended %sGB - Some models will not be downloaded\n" "$DISK_GB_ALLOCATED" "$DISK_GB_REQUIRED"
    fi
}

function provisioning_print_end() {
    printf "\nProvisioning complete:  Web UI will start now\n\n"
}

# Download from $1 URL to $2 file path, $3 optional filename
function provisioning_download() {
    local url="$1"
    local dir="$2"
    local custom_filename="$3"
    
    # For Civitai URLs, handle redirect and get filename from final URL
    if [[ "$url" == *"civitai.com"* ]]; then
        printf "Getting filename from Civitai redirect...\n"
        
        # Follow redirect and get the final URL
        local final_url=$(curl -sL -o /dev/null -w '%{url_effective}' "$url")
        
        # Extract filename from response-content-disposition parameter in the final URL
        local filename=$(echo "$final_url" | sed -n 's/.*filename%3D%22\([^%]*\)%22.*/\1/p')
        
        # If we couldn't extract filename, try alternative method
        if [[ -z "$filename" ]]; then
            # Try to get it from the redirect location header
            local redirect_url=$(curl -sI "$url" | grep -i "location:" | cut -d' ' -f2 | tr -d '\r')
            filename=$(echo "$redirect_url" | sed -n 's/.*filename%3D%22\([^%]*\)%22.*/\1/p')
        fi
        
        # If still no filename, use default
        if [[ -z "$filename" ]]; then
            filename="$(basename "$url").safetensors"
        fi
        if [[ -n "$custom_filename" ]]; then
            filename="$custom_filename"
        fi
        printf "Downloading as: %s\n" "$filename"
        wget -O "${dir}/${filename}" "$url"
    else
        if [[ -n "$custom_filename" ]]; then
            wget --header="Authorization: Bearer $HF_TOKEN" -qnc --content-disposition --show-progress -e dotbytes="${4:-4M}" -O "${dir}/${custom_filename}" "$url"
        else
            wget --header="Authorization: Bearer $HF_TOKEN" -qnc --content-disposition --show-progress -e dotbytes="${4:-4M}" -P "$dir" "$url"
        fi
    fi
}

provisioning_start
