FROM pytorch/pytorch
LABEL Name=ultimatetts Version=0.0.1

RUN apt-get -y update && apt install git libsndfile-dev libopenblas-dev -y
COPY requirements.txt .

# Install Montreal Forced Aligner
RUN conda create -n ultimate_tts -c conda-forge openblas python=3.7 openfst pynini ngram baumwelch 
SHELL ["conda", "run", "--no-capture-output", "-n", "ultimate_tts", "/bin/bash", "-c"]

# Install ultimate_tts requirements
RUN conda install pytorch==1.9 torchaudio==0.9 cudatoolkit=10.2 -c pytorch -y
RUN pip install --default-timeout=1000 -r requirements.txt 

RUN git clone https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner && cd ./Montreal-Forced-Aligner && pip3 install -r requirements.txt && python3 setup.py install
RUN mfa thirdparty download
RUN git clone https://github.com/kaldi-asr/kaldi ./kaldi
RUN mfa thirdparty kaldi ./kaldi
