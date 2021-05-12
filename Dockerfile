FROM pytorch/pytorch
LABEL Name=ultimatetts Version=0.0.1

RUN apt-get -y update && apt install git libsndfile-dev -y
COPY requirements.txt .

# Install Montreal Forced Aligner
RUN conda create -n ultimate_tts -c conda-forge openblas python=3.8 openfst pynini ngram baumwelch 
SHELL ["conda", "run", "-n", "ultimate_tts", "/bin/bash", "-c"]
RUN pip install montreal-forced-aligner
RUN mfa thirdparty download
RUN git clone https://github.com/kaldi-asr/kaldi ./kaldi
RUN mfa thirdparty kaldi ./kaldi


# Install ultimate_tts requirements
RUN conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
RUN pip install -r requirements.txt