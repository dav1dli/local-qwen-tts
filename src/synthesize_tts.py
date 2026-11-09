import argparse
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

PROCESS_STARTED_AT = datetime.now().astimezone()
PROCESS_START_MONOTONIC = time.perf_counter()

import numpy as np
import requests
import soundfile as sf
import torch
from huggingface_hub import configure_http_backend


def _insecure_hf_session():
    session = requests.Session()
    session.verify = False
    return session


configure_http_backend(backend_factory=_insecure_hf_session)

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_VOICE_NAME = "Otto"
SPEECH_RATE = 0.90
DEVICE_MAP = "mps"
DTYPE = torch.float16
MAX_CHARS_PER_CHUNK = 900
PAUSE_BETWEEN_CHUNKS_SECONDS = 0.25
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_AUDIO = PROJECT_ROOT / "voices" / "reference.wav"
REFERENCE_TEXT = PROJECT_ROOT / "voices" / "reference.txt"


def format_duration(seconds):
    total_seconds = max(0, int(round(seconds)))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    return f"{minutes} min {remaining_seconds} sec"


def split_text(text, max_chars=MAX_CHARS_PER_CHUNK):
    units = []
    paragraphs = [part for part in text.strip().split("\n\n") if part.strip()]
    for paragraph_index, paragraph in enumerate(paragraphs):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        for line_index, line in enumerate(lines):
            sentences = [part.strip() for part in re.split(r"(?<=[.!?…])\s+", line) if part.strip()]
            for sentence_index, sentence in enumerate(sentences):
                if not units:
                    separator = ""
                elif sentence_index > 0:
                    separator = " "
                elif line_index > 0:
                    separator = "\n"
                else:
                    separator = "\n\n"
                units.append((sentence, separator))

    pieces = []
    for unit, separator in units:
        unit_pieces = []
        if len(unit) <= max_chars:
            unit_pieces.append(unit)
        else:
            words = unit.split()
            piece = ""
            for word in words:
                candidate = f"{piece} {word}".strip()
                if len(candidate) > max_chars and piece:
                    unit_pieces.append(piece)
                    piece = word
                else:
                    piece = candidate
            if piece:
                unit_pieces.append(piece)
        for piece_index, piece in enumerate(unit_pieces):
            pieces.append((piece, separator if piece_index == 0 else " "))

    chunks = []
    current = ""
    for piece, separator in pieces:
        candidate = f"{current}{separator if current else ''}{piece}"
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def load_or_create_voice(model, voice_name):
    voice_profile = PROJECT_ROOT / "voices" / f"{voice_name}.pt"
    if voice_profile.exists():
        saved = torch.load(voice_profile, map_location="cpu", weights_only=True)
        if saved.get("model") != MODEL_ID:
            raise RuntimeError(
                f"{voice_profile} was created for {saved.get('model')}; "
                "remove it to recreate the profile for the configured model."
            )
        print(f"Using saved voice profile: {voice_name}")
        return [
            VoiceClonePromptItem(
                ref_code=saved["ref_code"],
                ref_spk_embedding=saved["ref_spk_embedding"],
                x_vector_only_mode=saved["x_vector_only_mode"],
                icl_mode=saved["icl_mode"],
                ref_text=saved["ref_text"],
            )
        ]

    if voice_name != DEFAULT_VOICE_NAME:
        raise FileNotFoundError(
            f"Voice profile not found: {voice_profile}. Create it with import_voice.py."
        )
    if not REFERENCE_AUDIO.is_file() or not REFERENCE_TEXT.is_file():
        raise FileNotFoundError(
            f"No saved {DEFAULT_VOICE_NAME} profile or source reference found. Expected "
            f"{REFERENCE_AUDIO} and {REFERENCE_TEXT}."
        )

    print(f"Creating saved voice profile: {DEFAULT_VOICE_NAME}")
    ref_text = REFERENCE_TEXT.read_text(encoding="utf-8").strip()
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(REFERENCE_AUDIO),
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    item = prompt[0]
    voice_profile.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": MODEL_ID,
            "ref_code": item.ref_code.detach().cpu() if item.ref_code is not None else None,
            "ref_spk_embedding": item.ref_spk_embedding.detach().cpu(),
            "x_vector_only_mode": item.x_vector_only_mode,
            "icl_mode": item.icl_mode,
            "ref_text": item.ref_text,
        },
        voice_profile,
    )
    return prompt


def ffmpeg_run(args):
    subprocess.run(args, check=True)


def main():
    print(f"Process started: {PROCESS_STARTED_AT.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    parser = argparse.ArgumentParser(
        description="Synthesize a .txt file with a saved voice and create WAV and MP3 outputs."
    )
    parser.add_argument("input", type=Path, help="Input text file")
    parser.add_argument(
        "--rate",
        type=float,
        default=SPEECH_RATE,
        help="Playback rate (default: 0.90; lower is slower, pitch is preserved)",
    )
    parser.add_argument("--language", default="German", help="Text language (default: German)")
    parser.add_argument("--voice", default=DEFAULT_VOICE_NAME, help="Saved voice profile name (default: Otto)")
    instruction_group = parser.add_mutually_exclusive_group()
    instruction_group.add_argument("--instruction", help="Optional global natural-language delivery instruction")
    instruction_group.add_argument("--instruction-file", type=Path, help="UTF-8 text file with a delivery instruction")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"input file not found: {input_path}")
    if input_path.suffix.lower() != ".txt":
        parser.error("input must be a .txt file")
    if not 0.5 <= args.rate <= 2.0:
        parser.error("--rate must be between 0.5 and 2.0 for FFmpeg atempo")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required; install it with: brew install ffmpeg")

    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        parser.error("input text file is empty")

    instruction = args.instruction
    if args.instruction_file is not None:
        instruction_path = args.instruction_file.expanduser().resolve()
        if not instruction_path.is_file():
            parser.error(f"instruction file not found: {instruction_path}")
        instruction = instruction_path.read_text(encoding="utf-8").strip()
        if not instruction:
            parser.error("instruction file is empty")

    output_stem = input_path.stem
    wav_path = input_path.with_name(f"{output_stem}.wav")
    mp3_path = input_path.with_name(f"{output_stem}.mp3")
    chunks = split_text(text)
    print(f"Input: {input_path}")
    print(f"Chunks: {len(chunks)} | Voice: {args.voice} | Rate: {args.rate:.2f}")
    print(f"Style instruction: {'enabled' if instruction else 'none'}")
    print("WARNING: Hugging Face TLS verification is disabled in this script for the Zscaler workaround.")
    print("Remove the _insecure_hf_session configuration after fixing the certificate trust.")

    print(f"Loading {MODEL_ID}...")
    model_setup_start = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map=DEVICE_MAP, dtype=DTYPE)
    voice_prompt = load_or_create_voice(model, args.voice)
    model_setup_time = time.perf_counter() - model_setup_start
    print(f"Model and voice setup: {format_duration(model_setup_time)}")

    with tempfile.TemporaryDirectory(prefix="qwen-tts-") as temp_dir:
        temp_path = Path(temp_dir)
        chunk_paths = []
        sample_rate = None
        generation_start = time.perf_counter()

        for index, chunk in enumerate(chunks, start=1):
            print(f"Generating chunk {index}/{len(chunks)}...")
            wavs, chunk_rate = model.generate_voice_clone(
                text=chunk,
                language=args.language,
                voice_clone_prompt=voice_prompt,
                instruct=instruction,
                do_sample=False,
                non_streaming_mode=True,
            )
            if sample_rate is None:
                sample_rate = chunk_rate
            elif chunk_rate != sample_rate:
                raise RuntimeError("The model returned inconsistent sample rates between chunks.")

            audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            pause = np.zeros(round(sample_rate * PAUSE_BETWEEN_CHUNKS_SECONDS), dtype=np.float32)
            chunk_path = temp_path / f"chunk-{index:05d}.wav"
            sf.write(chunk_path, np.concatenate((audio, pause)), sample_rate, subtype="PCM_24")
            chunk_paths.append(chunk_path)

        generation_time = time.perf_counter() - generation_start
        concat_list = temp_path / "chunks.txt"
        concat_list.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in chunk_paths),
            encoding="utf-8",
        )
        ffmpeg_run(
            [
                "ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list), "-filter:a", f"atempo={args.rate}",
                "-c:a", "pcm_s24le", str(wav_path),
            ]
        )

    ffmpeg_run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(wav_path),
            "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path),
        ]
    )
    audio_duration = sf.info(wav_path).duration
    total_process_time = time.perf_counter() - PROCESS_START_MONOTONIC
    print(f"WAV: {wav_path}")
    print(f"MP3: {mp3_path}")
    print(f"Generated audio length: {format_duration(audio_duration)}")
    print(f"TTS generation time: {format_duration(generation_time)}")
    print(f"Total process time: {format_duration(total_process_time)}")


if __name__ == "__main__":
    main()
