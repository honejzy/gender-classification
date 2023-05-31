FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04

# Install PyTorch
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip3 install -U scikit-learn
RUN pip3 install datasets deep_utils tensorboardX librosa mlflow clearml
RUN pip3 install --upgrade numba
RUN apt-get update && apt-get upgrade -y && apt-get install -y git && apt-get install -y libsndfile1

# Set up environment variables
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

RUN useradd -m d.soroka --uid=1008
USER d.soroka:1008  

# Set up a working directory
WORKDIR /
