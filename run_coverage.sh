#! /bin/bash


# IMG=nvcr.io/nvidia/pytorch:24.10-py3
IMG=monai


timestamp=$(date '+coverage_%Y%m%d_%H%M%S')
run_script="/tmp/coverage_${timestamp}.py"

cat - > $run_script << _EOF_
(
    # export CUDA_VISIBLE_DEVICES=0,1
    # export MAX_JOBS=64

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
    
    zip -r ${timestamp}.zip .coverage coverage.xml ${run_script}
    rm -rf .eggs monai.egg-info *.zarr .coverage cufile.log runner.log
    chown -R $(id -u):$(id -g) .
) 2>&1 | tee ${timestamp}.log
_EOF_


docker run -d --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v $(pwd):/monai -v /tmp:/tmp -w /monai --gpus '"device=0,1"' $IMG \
        /bin/bash $run_script 


# curl -Os https://uploader.codecov.io/latest/linux/codecov
# chmod +x codecov
# MONAI_HASH=$(python -c 'import monai; print(monai.__revision_id__)')
# CODECOV_TOKEN=go-find-it
# ./codecov -t ${CODECOV_TOKEN} -B dev -C ${MONAI_HASH} -f '!*.sh'
