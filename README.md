# Ultimate tts
Simple and reproducible framework for text to speech


## Quick start
1) Build docker container: `docker-compose build`
2) Place your data at ./downloads
2) Edit preprocessing and training params at config, for example `config/tts/tacotron2.yml`
3) Edit training pipeline at `train.sh`
4) Run training `./run_docker.sh`