FROM ubuntu:16.04

RUN apt-get update && apt-get -y install curl git wget
CMD ["/bin/bash"]

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install keyboard-configuration
RUN curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
RUN dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
RUN apt-get update && apt-get -y install cuda-9-0 --allow-unauthenticated

RUN wget http://archive.simtk.org/ktrprotocol/temp/cudnn-9.0-linux-x64-v7.1.tgz
RUN tar xvzf cudnn-9.0-linux-x64-v7.1.tgz
RUN cp cuda/lib64/* /usr/local/cuda/lib64/
RUN cp cuda/include/cudnn.h /usr/local/cuda/include/
RUN ldconfig /usr/local/cuda/lib64
RUN rm -rf cuda && rm cudnn-9.0-linux-x64-v7.1.tgz


# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN pip install --upgrade pip
RUN pip install tensorflow==1.8.0 tensorflow-gpu==1.8.0 keras==2.0.0 tifffile
