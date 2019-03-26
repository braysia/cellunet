#! /bin/bash
# The startup script for Google Compute Engine. Use Tesla K80 with Ubuntu 16.04.

if ! dpkg-query -W cuda-9-0; then
  # The 16.04 installer works with 16.10.
  apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
  curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  apt-get update
  apt-get install cuda-9-0 -y
fi
# Enable persistence mode
nvidia-smi -pm 1

# CuDNN installation
if [ ! -f /usr/local/cuda-9.0/include/cudnn.h ]; then # FIXME
  wget http://archive.simtk.org/ktrprotocol/temp/cudnn-9.0-linux-x64-v7.1.tgz
  tar xvzf cudnn-9.0-linux-x64-v7.1.tgz
  sudo cp cuda/lib64/* /usr/local/cuda/lib64/
  sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
  sudo ldconfig /usr/local/cuda/lib64
  rm -rf cuda
  rm cudnn-9.0-linux-x64-v7.1.tgz
fi

# install pyenv
if ! [ -x "$(command -v pyenv)" ]; then  # FIXME
  git clone https://github.com/pyenv/pyenv.git ~/.pyenv
  echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
  echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
  echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc
  exec "$SHELL"
  pyenv install anaconda2-5.2.0
  pyenv global anaconda2-5.2.0
  pip install tensorflow==1.8.0 tensorflow-gpu==1.8.0 keras==2.0.0 tifffile
fi
