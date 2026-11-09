# Local Qwen3-TTS on macOS

This workspace contains the upstream Qwen3-TTS source in `Qwen3-TTS/`, local scripts are in `src/`. They import cloned voices and synthesizing text.

Why? This process produces fairly natural sounding speech, especially in comparison with Azure Speech (which does not process the text semantically) which can be combined with the [Easy German Reading](https://github.com/dav1dli/easygerman) to synthesize audio texts.

Imported high quality human or synthetic voices makes the output noticeably better in comparison with model's built-in voices.  

The scripts use the upstream PyTorch implementation (`Qwen3TTSModel`) with Apple Metal (MPS) on Apple Silicon. This is not the separate MLX implementation.

## Requirements

- macOS with Apple Silicon is recommended. MPS should be available in the installed PyTorch.
- Python 3.11 or 3.12 is recommended.
- Internet access for the first model download. The Base model weights are several gigabytes; leave sufficient disk space and memory.
- A clean voice recording and a transcript that matches the recording.

Check Python and Mac architecture:

```bash
python3 --version
uname -m
```

On Apple Silicon, `uname -m` should normally print `arm64`.

### Numbers

On Apple M2 (2022) with 24GB RAM the ratio is 1.67 (i.e. 15min sound file is synthesized in 25min).

On Intel Xeon E5 1630 3.7GHz 4 cores, 32GB RAM, NVidia M2000 GPU 4GB Linux the ratio is 5.8 (i.e. with the overhead of model loading it takes 6 min to generate 1min audio). The GPU cannot be used at all, because it is too old and because the memory is too small, so the process can work on the CPU only.

## Get the Source with Git

Check whether Git is installed:

```bash
git --version
```

If the command is unavailable, install Apple's Command Line Tools, which include Git:

```bash
xcode-select --install
```

For a fresh workspace, run the clone command from the `qwen-tts/` workspace root:

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git Qwen3-TTS
```

This checks out the official source into `Qwen3-TTS/`. Skip the clone if that directory already exists. The remaining setup commands below assume the workspace layout shown above.

## Create the Environment and Install Qwen3-TTS

Run these commands from the workspace root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
cd Qwen3-TTS
python -m pip install -e .
```

The editable install uses the checked-out source and installs its Python dependencies. To activate the environment in a new terminal, run:

```bash
source .venv/bin/activate
```

Verify that this PyTorch installation sees Apple GPU support:

```bash
python -c "import platform, torch; print('Architecture:', platform.machine()); print('MPS available:', torch.backends.mps.is_available())"
```

For Apple Silicon, expect `arm64` and `MPS available: True`. The test script currently selects MPS and uses `torch.float16`.

## Clone a Voice

Prepare two matching files:

- `reference.wav`: a clear recording of the voice to clone. Use one speaker, with little background noise, music, reverberation, or processing. A few clean seconds can work; a longer, natural sample gives the model more voice and prosody to condition on.
- `reference.txt`: a plain-text, verbatim transcript of the speech in that WAV. Keep the language and word order accurate; do not include Markdown, speaker labels, or notes that are not spoken.

The transcript is required for the full reference-conditioned cloning mode used here. It is not the new text you want synthesized. Use audio you have permission to clone.

```bash
python src/import_voice.py voices/reference.wav voices/reference.txt --name Otto
```

This writes `voices/Otto.pt` in the workspace root. The profile contains the reference speech codes and speaker embedding, not the model weights. The model weights download on the first run and are cached locally; the model still loads into memory for each new Python process. If the selected profile does not exist, synthesis can create the default `Otto` profile from the workspace's `reference.wav` and `reference.txt` automatically.

## Generate Speech

Prepare a plain-text input named `name-TTS.txt`. The `-TTS` suffix distinguishes input text from the generated audio. Preserve paragraphs and normal dialogue punctuation; the Base clone model speaks the entire text in the selected cloned voice.

Select the saved voice and synthesize:

```bash
python src/synthesize_tts.py path/to/name-TTS.txt --voice Otto
```

The script chunks long text, then writes `name.wav` and `name.mp3` beside the input file. It uses a pitch-preserving 90% playback rate by default. Override the rate with `--rate 0.95` (closer to normal speed) or `--rate 0.85` (slower).

### Delivery Instructions

There are no documented inline emotion or speaker tags for the Base voice-clone model. Do not add labels such as `[angry]`, `[whisper]`, or `<emotion=happy>` to the text; they may be spoken literally. Dialogue punctuation and paragraph breaks can guide pauses, but do not switch the cloned voice between characters.

The converter also accepts one global natural-language delivery instruction, applied to each text chunk. Supply it directly:

```bash
python src/synthesize_tts.py path/to/name-TTS.txt --voice Otto \
    --instruction "Lies auf Deutsch wie ein ruhiger, warmherziger Erzähler. Sprich klar und bedächtig. Gestalte Dialoge lebendig, aber bleib bei derselben Stimme."
```

Or put the instruction in a UTF-8 text file and pass its path:

```bash
python src/synthesize_tts.py .path/to/name-TTS.txt --voice Otto \
    --instruction-file instruction.txt
```

The underlying Qwen model accepts a separate instruction-token input, but the official model documentation does not promise instruction control for Base cloning. This optional path is experimental: test it on a short passage first, and expect the model to follow the instruction inconsistently or not at all. The prompt is separate from the spoken text; it is not an inline tag system.

## Select a Smaller Model

If the 1.7B model does not fit available unified memory, change `MODEL_ID` in both `import_voice.py` and `synthesize_tts.py` to:

```python
MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
```

The 0.6B Base model supports voice cloning as well. Its weights will be downloaded and cached on first use.

## Intel Macs or MPS Problems

If MPS is unavailable, or PyTorch reports an unsupported MPS operation, use CPU as a slower fallback. In both `import_voice.py` and `synthesize_tts.py`, set `DEVICE_MAP = "cpu"` and `DTYPE = torch.float32`. Re-import a voice profile after changing the model ID; profiles are tied to the model that created them. CPU generation can be substantially slower.

## Zscaler Certificate Workaround

The local test, voice-import, and synthesis scripts currently configure a temporary Hugging Face HTTP session with TLS certificate verification disabled. This is only to get model downloads working through the current Zscaler certificate issue; it does not change macOS trust settings or disable verification for other programs. Disabling verification makes downloads vulnerable to interception, so use it only on a trusted network and remove it once the Zscaler root certificate is installed in the Python certificate trust chain.

The workaround is near the imports in each script: it imports `requests` and `configure_http_backend`, defines `_insecure_hf_session()` with `session.verify = False`, and registers that session. To restore normal verification, remove the corresponding import, function, and `configure_http_backend(...)` call from the script you are using. Command-line Hugging Face downloads do not automatically inherit the scripts' session.

## Troubleshooting

- **Avoid Hugging Face checks after caching the model**: set `HF_HUB_OFFLINE=1` for a run to prevent Hub metadata requests. This works only after all required model files have been downloaded. From `Qwen3-TTS/`, for example:

    ```bash
    HF_HUB_OFFLINE=1 python ../src/synthesize_tts.py path/to/name-TTS.txt --voice Otto
    ```

- **Hugging Face SSL certificate error**: confirm the temporary workaround is present in the script you are running. For a lasting fix, add the organization’s Zscaler root certificate to the certificate bundle used by Python, then remove the workaround.
- **MPS unavailable**: confirm you are using an Apple Silicon Python installation and a current PyTorch build; otherwise use the CPU fallback above.
- **Out of memory**: try the 0.6B Base model and close other memory-heavy applications.
- **Wrong voice or content**: verify that `reference.txt` is an accurate transcript of `reference.wav` and that the reference recording is clear.

For upstream implementation details and other model examples, see [`Qwen3-TTS/README.md`](Qwen3-TTS/README.md).
