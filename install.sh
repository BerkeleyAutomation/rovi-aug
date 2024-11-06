#!/bin/bash

# Conda environment variable names
CONDA_SAM_ENV_NAME="rlds_env_sam"
CONDA_R2R_ENV_NAME="rlds_env_r2r"
CONDA_VIDEO_INPAINT_ENV_NAME="rlds_env_video_inpaint"
CONDA_ZERO_NVS_ENV_NAME="rlds_env_zeronvs"

# Print introductory information
echo "Welcome to the Rovi-Aug Installer!"
echo "This script will help you install the necessary components for Rovi-Aug."

eval "$(conda shell.bash hook)"
mkdir deps/
mkdir weights/

# Query the user for installation
read -p "Do you want to proceed with installing the robot segmentation code? (Required for Ro-Aug) (y/n): " response

install_common_dependencies () {
    [ ! -d "deps/rlds_dataset_mod" ] && git clone https://github.com/kpertsch/rlds_dataset_mod.git deps/rlds_dataset_mod
    pip install -e deps/rlds_dataset_mod/
    [ ! -d "deps/dlimp" ] && git clone https://github.com/kvablack/dlimp.git deps/dlimp
    pip install -e deps/dlimp/
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install -e .
}

# Check the user's response
if [[ "$response" == "y" || "$response" == "Y" ]]; then
    echo "Cloning robot segmentation code..."
    # Clone the repository
    git clone --branch SAM https://github.com/chenfengxu714/diffusers-robotic-inpainting.git deps/SAMed_h

    echo "Creating conda environment for pipeline named $CONDA_SAM_ENV_NAME..."
    # Create a conda environment
    conda create -n $CONDA_SAM_ENV_NAME python=3.9
    conda activate $CONDA_SAM_ENV_NAME
    pip install -r deps/SAMed_h/requirements.txt
    pip install icecream
    pip install safetensors
    pip install gdown
    install_common_dependencies

    echo "Downloading weights for SAM..."
    mkdir weights/mask

    # Download the lora weights
    # The output weights in mask will be named as source_robot.pth
    gdown --fuzzy https://drive.google.com/file/d/1ManvjY52QiMt5KhJhpM8APJ9WUCvtE1L/view?usp=sharing -O weights/mask/franka.pth

    # Download SAM weights
    gdown --fuzzy https://drive.google.com/file/d/13j4pGfH0kF5YCmmuqs3-tS88kIA9cwEn/view?usp=sharing -O weights/mask/sam_vit_h_4b8939.pth
    conda deactivate
else
    echo "Robot segmentation installation aborted by the user."
fi

# Query the user for installation
read -p "Do you want to proceed with installing the robot to robot diffusion code? (Required for Ro-Aug) (y/n): " response

# Check the user's response
if [[ "$response" == "y" || "$response" == "Y" ]]; then
    echo "Cloning the robot to robot diffusion code..."
    # Clone the repository
    git clone --branch r2r https://github.com/chenfengxu714/diffusers-robotic-inpainting.git deps/r2r

    echo "Creating conda environment for pipeline named $CONDA_R2R_ENV_NAME..."
    # Create a conda environment
    conda create -n $CONDA_R2R_ENV_NAME python=3.9
    conda activate $CONDA_R2R_ENV_NAME
    pip install -r deps/r2r/requirements.txt
    pip install gdown
    install_common_dependencies

    mkdir weights/r2r

    # Download control-net weights
    gdown --fuzzy https://drive.google.com/drive/folders/124ZxGqmNLsMhbKDRX1SifT5IGDtoBEWY?usp=sharing --folder -O weights/r2r/franka_to_ur5

    conda deactivate
else
    echo "Robot to robot installation aborted by the user."
fi

# Query the user for installation
read -p "Do you want to proceed with installing the video inpainting code? (Required for Ro-Aug) (y/n): " response

# Check the user's response
if [[ "$response" == "y" || "$response" == "Y" ]]; then
    echo "Cloning video inpainting code..."
    # Clone the repository
    git clone --branch video-inpaint https://github.com/chenfengxu714/diffusers-robotic-inpainting.git deps/video-inpaint

    echo "Creating conda environment for pipeline named $CONDA_VIDEO_INPAINT_ENV_NAME..."
    # Create a conda environment
    conda env create -f deps/video-inpaint/environment.yml -n $CONDA_VIDEO_INPAINT_ENV_NAME
    conda activate $CONDA_VIDEO_INPAINT_ENV_NAME
    pip install gdown
    install_common_dependencies

    mkdir weights/video-inpaint
    gdown --fuzzy https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3/view?usp=sharing -O weights/video-inpaint/E2FGVI-HQ-CVPR22.pth
    conda deactivate
else
    echo "Video inpainting installation aborted by the user."
fi

# Query the user for installation
read -p "Do you want to proceed with installing the viewpoint augmentation code? (Required for /Vi-Aug) (y/n): " response

# Check the user's response
if [[ "$response" == "y" || "$response" == "Y" ]]; then
    echo "Cloning viewpoint augmentation code..."
    # Clone the repository
    git clone --recurse-submodules https://github.com/kylesargent/ZeroNVS.git deps/ZeroNVS
    
    echo "Creating conda environment and installing deps for pipeline named $CONDA_ZERO_NVS_ENV_NAME..."
    # Create a conda environment, instructions taken directly from the ZeroNVS README
    conda create -n $CONDA_ZERO_NVS_ENV_NAME python=3.8 pip
    conda activate $CONDA_ZERO_NVS_ENV_NAME

    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
    pip install ninja pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

    pip install -r deps/ZeroNVS/requirements-zeronvs.txt
    pip install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.0.0_cu118.html

    pip install -e deps/ZeroNVS/zeronvs_diffusion/zero123

    install_common_dependencies

    echo "Downloading weights for ZeroNVS..."
    # Download the weights
    pip install gdown
    mkdir weights/zeronvs
    gdown --fuzzy https://drive.google.com/file/d/17WEMfs2HABJcdf4JmuIM3ti0uz37lSZg/view?usp=sharing -O weights/zeronvs/zeronvs.ckpt
    conda deactivate
else
    echo "Viewpoint augmentation installation aborted by the user."
fi

