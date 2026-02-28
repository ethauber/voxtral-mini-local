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

Streaming speech-to-text (v3 - VAD-based)

`streaming-transcribe.py` provides continuous, toggle-able transcription using Voice Activity Detection (VAD) — speak naturally, and the system detects when you pause to transcribe complete phrases.

```bash
# start streaming with defaults (0.5s silence threshold)
python streaming-transcribe.py

# adjust silence threshold for faster/slower speakers
python streaming-transcribe.py --silence-threshold 0.3  # faster
python streaming-transcribe.py --silence-threshold 0.8  # slower

# save transcripts to file
python streaming-transcribe.py -o transcripts.log

# verbose mode with VAD debug info
python streaming-transcribe.py -v

# combine options
python streaming-transcribe.py --silence-threshold 0.4 -o session.log -v
```

**Controls:**
- `SPACE`: toggle recording on/off
- `ESC`: stop and exit

**How it works (VAD-based):**
- Uses Silero VAD to detect speech vs. silence in real-time
- Accumulates audio while you speak
- When silence is detected (default 0.5s), sends the complete phrase for transcription
- **No word cutting**: phrases end at natural pauses, so no overlapping or deduplication needed
- Model loads once at startup (includes warmup to reduce first-chunk latency)

**VAD Parameters:**
- `--silence-threshold`: duration of silence to end a phrase (default 0.5s)
- `--min-speech-duration`: minimum phrase length to transcribe (default 0.3s, filters noise)
- `--max-phrase-duration`: safety limit for long phrases (default 30s)

**Requirements:**
- `torch` and Silero VAD (added to `reqs.in`)
- `pynput` for keyboard control
- Same Voxtral model and dependencies as other scripts
- **macOS users**: Grant accessibility permissions when prompted for keyboard monitoring (System Settings → Privacy & Security → Accessibility)

**Benefits of VAD approach:**
- Natural phrase boundaries improve accuracy
- No repeated words or text deduplication
- Adapts to speaker pace automatically
- Filters out background noise bursts
- Filters out background noise bursts

Notes

- Virtual environment: `.venv-v2` (`python -m venv .venv-v2 && source .venv-v2/bin/activate`).
- `transformers` is pinned to a specific commit via a VCS URL in `reqs.in` to match compatibility used when this project was tested.
- The `hello-voxtral.py` fix is minimal and targeted — if you run into a different `TranscriptionInputs` shape, please paste the traceback and I will adapt the conversion.
