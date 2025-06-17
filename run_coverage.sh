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


run_script=$(date '+/tmp/coverage_%s.py')
timestamp=$(date '+coverage_%Y%m%d_%H%M%S')

cat - > $run_script << _EOF_
(
    pwd
    nvidia-smi

    export DEBIAN_FRONTEND=noninteractive
    apt-get update 
    apt-get install -y libopenslide0 curl git zip

    python -m pip install --upgrade pip
    pip uninstall monai -y
    pip install -r requirements-dev.txt
    pip install .[all]
    
    #BUILD_MONAI=1 ./runtests.sh --build --coverage --unittests --net --disttests 
    BUILD_MONAI=1 ./runtests.sh --build --coverage --unittests 
    BUILD_MONAI=1 ./runtests.sh --build --coverage --net
    BUILD_MONAI=1 ./runtests.sh --build --coverage --disttests
    coverage xml --ignore-errors
    
    zip -r ${timestamp}.zip .coverage coverage.xml
    chown -R $(id -u):$(id -g) .
) 2>&1 | tee ${timestamp}.log
_EOF_


docker run -d --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v $(pwd):/monai -v /tmp:/tmp -w /monai --gpus '"device=0,1"' $IMG \
        /bin/bash $run_script 
        #2>&1 | tee run_coverage.log
