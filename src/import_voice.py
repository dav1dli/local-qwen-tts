import argparse
from pathlib import Path

import requests
import torch
from huggingface_hub import configure_http_backend


def _insecure_hf_session():
    session = requests.Session()
    session.verify = False
    return session


configure_http_backend(backend_factory=_insecure_hf_session)

from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEVICE_MAP = "mps"
DTYPE = torch.float16
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Create a reusable Qwen3-TTS voice profile from a WAV and matching transcript."
    )
    parser.add_argument("audio", type=Path, help="Reference voice recording (.wav)")
    parser.add_argument("transcript", type=Path, help="Exact transcript of the reference recording (.txt)")
    parser.add_argument("--name", required=True, help="Voice profile name, without the .pt suffix")
    args = parser.parse_args()

    audio_path = args.audio.expanduser().resolve()
    transcript_path = args.transcript.expanduser().resolve()
    if not audio_path.is_file():
        parser.error(f"audio file not found: {audio_path}")
    if audio_path.suffix.lower() != ".wav":
        parser.error("reference audio must be a .wav file")
    if not transcript_path.is_file():
        parser.error(f"transcript file not found: {transcript_path}")
    if transcript_path.suffix.lower() != ".txt":
        parser.error("reference transcript must be a .txt file")
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("--name must be a simple filename stem, such as Otto")

    ref_text = transcript_path.read_text(encoding="utf-8").strip()
    if not ref_text:
        parser.error("reference transcript is empty")

    profile_path = PROJECT_ROOT / "voices" / f"{args.name}.pt"
    if profile_path.exists():
        parser.error(f"profile already exists: {profile_path}; choose another name or remove it first")

    print("WARNING: Hugging Face TLS verification is disabled in this script for the Zscaler workaround.")
    print("Remove the _insecure_hf_session configuration after fixing the certificate trust.")
    print(f"Loading {MODEL_ID}...")
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map=DEVICE_MAP, dtype=DTYPE)

    print(f"Creating voice profile {args.name} from {audio_path.name}...")
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=str(audio_path),
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    item = prompt_items[0]

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": MODEL_ID,
            "ref_code": item.ref_code.detach().cpu() if item.ref_code is not None else None,
            "ref_spk_embedding": item.ref_spk_embedding.detach().cpu(),
            "x_vector_only_mode": item.x_vector_only_mode,
            "icl_mode": item.icl_mode,
            "ref_text": item.ref_text,
        },
        profile_path,
    )
    print(f"Saved reusable voice profile: {profile_path}")


if __name__ == "__main__":
    main()
