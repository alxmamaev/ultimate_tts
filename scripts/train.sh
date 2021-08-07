cd $ULTIMATE_TTS_DIR

echo "Extracting features..."
PYTHONPATH=. python3 ./scripts/preprocess_dataset.py ./config/tts/tacotron2.yml

# echo "Starting tts training..."
# catalyst-dl run --config ./config/tts/tacotron2.yml

# echo "Starting vocoder training..."

# echo "Starting prosody predictor training..."