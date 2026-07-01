#!/bin/bash

ARM64_GCC_DIR=

ARM_GCC_DIR=

GCC_DIR=

MINGW_GCC_DIR=

ROOT_DIR=/home/ysr/project/c


if [ -z "$1" ];then
    export ARCH=$(uname -m)
    echo "set default env $ARCH"
    export PLATFORM=
else
    if [ $1 == "aarch64" ]
    then
        echo "set aarch64 env"
        export PLATFORM=aarch64-linux-gnu
        export ARCH=aarch64
        export PATH=${PATH}:${ARM64_GCC_DIR}/bin

    elif [ $1 == "arm" ]
    then
        echo "set arm env"
        export PLATFORM=arm-linux-gnueabihf
        export ARCH=arm
        export PATH=${PATH}:${ARM_GCC_DIR}/bin
    else
        echo "set default env"
        export PLATFORM=
        export ARCH=x64
    fi
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOT_DIR/3rdparty/ggml/$ARCH/lib
