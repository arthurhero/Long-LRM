python -m pip install --upgrade pip
pip install packaging
pip install torch==2.1.2 torchvision==0.16.2
pip install Pillow==10.4.0 opencv-python==4.10.0.84 numpy==1.24.4 scikit-image==0.21.0 matplotlib==3.7.5 scikit-learn==1.3.2
pip install easydict==1.13 kornia==0.7.3 pyyaml==6.0.2 wandb==0.19.1 einops==0.8.0 lpips==0.1.4 jaxtyping==0.2.19 termcolor==2.4.0
pip install xformers==0.0.23.post1
sudo apt update && sudo apt install -y ffmpeg
pip install videoio==0.3.0 ffmpeg-python==0.2.0
pip install causal-conv1d==1.4.0
pip install mamba-ssm==2.2.2
pip install git+https://github.com/nerfstudio-project/gsplat
# echo "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7" >> ~/.bash_profile # might need this line for gsplat