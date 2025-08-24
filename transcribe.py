"""
Transcribe audio using OpenAI Whisper (local model).
Usage:
    python transcribe.py --input data/uploaded_meeting.mp3 --output data/transcript.json

Or import:
    from transcribe import transcribe
    transcribe("data/uploaded_meeting.mp3", "data/transcript.json")
"""

import argparse, json, os


def transcribe(input_path: str, output_path: str, model_size: str = "base"):
    """
    Transcribe audio using whisper.
    Args:
        input_path: path to input audio file
        output_path: path to save JSON transcript
        model_size: whisper model size (tiny, base, small, medium, large)
    """
    try:
        import whisper
    except ImportError:
        raise RuntimeError("Please install whisper: pip install -U openai-whisper")

    model = whisper.load_model(model_size)
    result = model.transcribe(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved transcript to {output_path}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to input audio (e.g. mp3, wav, m4a)")
    p.add_argument("--output", required=True, help="Path to save transcript JSON")
    p.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    args = p.parse_args()

    transcribe(args.input, args.output, model_size=args.model)
