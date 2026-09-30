#!/usr/bin/env bash
# Raspberry Pi 4 (4 GB) one-shot setup + self-test.
# Usage:  ./setup_pi.sh            (installs deps + runs a file-mode self-test)
#         ./setup_pi.sh --mic      (then launches mic + GUI)
set -e
cd "$(dirname "$0")"

echo "==> System packages"
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip portaudio19-dev \
                        libsndfile1 ffmpeg

echo "==> Virtual environment"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Self-test (file mode: prints transcript, then the intent)"
python app.py --mode file \
  --file test_clips/commands/PLAY_MUSIC_s99_v3_clean.wav

if [[ "${1:-}" == "--mic" ]]; then
  echo "==> Launching mic + GUI - say 'Hey Rapi'"
  export DISPLAY="${DISPLAY:-:0}"
  exec python app.py --mode mic --gui
fi

echo "Done. Next:  ./setup_pi.sh --mic   (or: python app.py --mode mic)"
echo "Full guide:  docs/raspberry_pi.md"
