export HF_TOKEN=your_huggingface_token_here
```bash
export HF_TOKEN=your_huggingface_token_here
```

Quick run

1. Ensure `hello.wav` is present in the repo root.
2. Run the example script:

```bash
python hello-voxtral.py
```

Why this change to `hello-voxtral.py`

- Fix: the script converts the transcription inputs into a plain mapping before unpacking into `model.generate(...)`. This avoids the TypeError when the library returns a `TranscriptionInputs` object not directly unpackable with `**`.

Dependencies and workflows

- Use `reqs.in` as the canonical top-level dependency list (this repo keeps `reqs.in` for that purpose).
- To produce a pinned `requirements.txt` from `reqs.in` (recommended):

```bash
pip install pip-tools
pip-compile reqs.in --output-file requirements.txt
```

- To apply the pinned requirements into the environment:

```bash
pip install -r requirements.txt
```

- If you want a full frozen snapshot of the current environment for sharing or deployment:

```bash
pip freeze > requirements.full.txt
```

Python version

- Recommended: Python 3.11.4. Use `pyenv` to manage the version:

```bash
pyenv install 3.11.4
pyenv local 3.11.4
```

PyTorch and platform-specific wheels

- Install `torch` separately using the official instructions at https://pytorch.org for your macOS configuration; example:

```bash
pip install torch
```

Record and transcribe (v2)

`record-and-transcribe.py` records audio from your mic and feeds it straight into Voxtral for transcription — no pre-existing WAV needed.

```bash
# record 5 seconds (default) and transcribe
python record-and-transcribe.py

# record 10 seconds, save to custom path
python record-and-transcribe.py -d 10 -o my_clip.wav
```

- Requires `sounddevice` (added to `reqs.in`; uses PortAudio under the hood).
- Records 16 kHz mono WAV to match what Voxtral expects.
- macOS will prompt for microphone permission on first run.

Streaming speech-to-text (v3)

`streaming-transcribe.py` provides continuous, toggle-able transcription — speak whenever you want, pause/resume with a hotkey.

```bash
# start streaming with default 5s chunks
python streaming-transcribe.py

# use 10s chunks, save transcripts to file
python streaming-transcribe.py -d 10 -o transcripts.log
```

**Controls:**
- `SPACE`: toggle recording on/off
- `ESC`: stop and exit

**How it works:**
- Audio capture runs continuously in background thread
- Chunks are queued and transcribed in parallel
- Model loads once at startup (includes warmup to reduce first-chunk latency)
- Real-time timestamps show when each transcript was captured

**Requirements:**
- `pynput` for keyboard control (added to `reqs.in`)
- Same Voxtral model and dependencies as other scripts

Notes

- Virtual environment: `.venv-v2` (`python -m venv .venv-v2 && source .venv-v2/bin/activate`).
- `transformers` is pinned to a specific commit via a VCS URL in `reqs.in` to match compatibility used when this project was tested.
- The `hello-voxtral.py` fix is minimal and targeted — if you run into a different `TranscriptionInputs` shape, please paste the traceback and I will adapt the conversion.
