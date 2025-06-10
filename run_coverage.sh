#! /bin/bash

# docker build . -f ../tests/jenkins/Dockerfile -t jenkins

# pip uninstall monai -y

export CUDA_VISIBLE_DEVICES=0,1
export MAX_JOBS=64

rm -rf .eggs monai.egg-info *.zarr .coverage cufile.log runner.log

# BUILD_MONAI=1 ./runtests.sh --build --coverage --unittests --disttests
# BUILD_MONAI=1 ./runtests.sh --build --coverage --net 

# BUILD_MONAI=1 ./runtests.sh -u --net --coverage

# coverage combine --append .coverage

# IMG=nvcr.io/nvidia/pytorch:24.10-py3
IMG=monai

(
docker run -i --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v $(pwd):/monai -w /monai --gpus '"device=0,1"' $IMG /bin/bash -s << _EOF_

    pwd
    nvidia-smi

    # export DEBIAN_FRONTEND=noninteractive
    # apt-get update 
    # apt-get install -y libopenslide0 curl git zip

    # python -m pip install --upgrade pip
    # pip uninstall monai -y
    # pip install -r requirements-dev.txt
    # pip install .[all]
    
    #BUILD_MONAI=1 ./runtests.sh --build --coverage --unittests --net --disttests 
    BUILD_MONAI=1 ./runtests.sh --build --coverage --unittests 
    BUILD_MONAI=1 ./runtests.sh --build --coverage --net
    coverage xml --ignore-errors
    
    zip -r $(date '+coverage_%Y%m%d_%H%M%S.zip') .coverage coverage.xml
    chown -R $(id -u):$(id -g) .
_EOF_

) 2>&1 | tee run_coverage.log