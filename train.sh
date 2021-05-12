echo "Extracting features..."
cd /ultimate_tts
PYTHONPATH=. python3 ./tools/preprocess_dataset.py ./config/tts/tacotron2.yml

echo "Starting tts training..."

echo "Starting vocoder training..."

echo "Starting prosody predictor training..."