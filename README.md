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

Notes

- `transformers` is pinned to a specific commit via a VCS URL in `reqs.in` to match compatibility used when this project was tested.
- The `hello-voxtral.py` fix is minimal and targeted — if you run into a different `TranscriptionInputs` shape, please paste the traceback and I will adapt the conversion.
