#!/bin/bash

if [ "$1" != "cpu" ] && [ "$1" != "cpu" ]; then
    echo "device may be gpu or cpu"
else
    if [ "$2" = "train" ]; then
        docker-compose run ultimate_tts_$1 conda run --no-capture-output -n ultimate_tts /bin/sh "/ultimate_tts/scripts/train.sh"
    elif [ "$2" = "dev" ]; then
        docker-compose run ultimate_tts_$1 conda run --no-capture-output -n ultimate_tts /bin/bash
    else
        echo "Running type may be train or dev"
    fi
fi