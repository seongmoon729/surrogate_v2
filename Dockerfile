# Written by Seongmoon Jeong - 2022.08.22

FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# Temporarily added for issue.
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Use kakao mirror
RUN sed -i 's@archive.ubuntu.com@mirror.kakao.com@g' /etc/apt/sources.list

# Set an environment variable.
ENV DEBIAN_FRONTEND noninteractive

# Install linux programs.
RUN apt-get update && apt-get install -y \
    software-properties-common \
    ca-certificates lsb-release git \
    protobuf-compiler ninja-build \
    tree curl wget nano vim htop screen rsync default-jdk 

# Install latest cmake.
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
    gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
RUN apt-get update
RUN apt-get install -y kitware-archive-keyring
RUN rm /etc/apt/trusted.gpg.d/kitware.gpg
RUN apt-get install -y cmake

# Install python3.7 & deps.
RUN apt-get install -y \
    python3.7-dev python3-pip python3-opencv python3-lxml
RUN ln -sv /usr/bin/python3.7 /usr/bin/python
RUN python -m pip install -U pip 

# Install pytorch & detectron2.
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="8.0"

# Torch==1.9.0 & Detectron2==0.5
# RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f \
#     "https://download.pytorch.org/whl/torch_stable.html"
# RUN pip install detectron2==0.5 -f \
#     "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"

# Torch==1.10.0 & Detectron2==0.6
RUN pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f \
    "https://download.pytorch.org/whl/torch_stable.html"
RUN pip install detectron2==0.6 -f \
    "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html"

# Install object_dection API.
RUN pip install tensorflow-cpu==2.8.1
WORKDIR /root
RUN git clone https://github.com/tensorflow/models.git
RUN cd /root/models/research/ && protoc object_detection/protos/*.proto --python_out=.
WORKDIR /root/models/research
RUN cp object_detection/packages/tf2/setup.py .
RUN pip install --no-deps .

# Install others.
RUN python -m pip install \
    parmap scikit-image seaborn ray[default]==2.0.0 compressai fiftyone
ENV TZ=Asia/Seoul

# Install FFmpeg
RUN add-apt-repository ppa:savoury1/ffmpeg4
RUN apt-get update
RUN apt-get install -y ffmpeg

# Install VTM
ENV VTM_VERSION=12.0
WORKDIR /root
RUN git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git
RUN cd VVCSoftware_VTM && git checkout tags/VTM-${VTM_VERSION} && mkdir build && \
    cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
RUN mv /root/VVCSoftware_VTM/bin/EncoderAppStatic /usr/bin/vtm
RUN mv /root/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg /usr/local/etc/

# Install VVenC & VVdeC.
ENV VVC_VERSION=1.4.0
WORKDIR /root
RUN git clone https://github.com/fraunhoferhhi/vvenc.git
RUN cd vvenc && git checkout tags/v${VVC_VERSION} && make install-release
RUN mv /root/vvenc/bin/release-static/vvenc* /usr/bin/
WORKDIR /root
RUN git clone https://github.com/fraunhoferhhi/vvdec.git
RUN cd vvdec && git checkout tags/v${VVC_VERSION} && make install-release
RUN mv /root/vvdec/bin/release-static/vvdec* /usr/bin/

# Set working directory.
WORKDIR /surrogate_v2
