cd /ultimate_tts

echo "Extracting features..."
PYTHONPATH=. python3 ./tools/preprocess_dataset.py ./config/tts/tacotron2.yml

head -n -300 ./data/metadata.csv > ./data/train_metadata.csv
tail -n 300 ./data/metadata.csv > ./data/valid_metadata.csv

echo "Starting tts training..."
catalyst-dl run --config ./config/tts/tacotron2.yml

echo "Starting vocoder training..."

echo "Starting prosody predictor training..."