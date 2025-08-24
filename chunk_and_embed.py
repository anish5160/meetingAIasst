"""
Chunk a transcript JSON (from whisper) and create embeddings using sentence-transformers.
Saves embeddings (.npy) and metadata JSON (chunks.json).

Usage (CLI):
    python chunk_and_embed.py --input data/transcript.json --output data/index

Or import:
    from chunk_and_embed import chunk_and_embed
    chunk_and_embed("data/transcript.json", "data/index")
"""

import argparse, json, os
from tqdm import tqdm

CHUNK_SIZE = 400  # characters per chunk
OVERLAP = 100     # characters overlap between chunks


def load_transcript(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping character chunks."""
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks


def chunk_and_embed(input_path: str, output_dir: str):
    """Load transcript, chunk it, create embeddings, save to output_dir."""
    data = load_transcript(input_path)
    text = data.get("text") or " ".join(
        s.get("text", "") for s in data.get("segments", [])
    )
    if not text.strip():
        raise RuntimeError("Transcript is empty or invalid.")

    chunks = chunk_text(text)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError("Please install: pip install -U sentence-transformers")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)

    # Save embeddings + chunks
    os.makedirs(output_dir, exist_ok=True)
    import numpy as np
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(chunks)} chunks + embeddings to {output_dir}")
    return chunks, embeddings


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to transcript.json")
    p.add_argument("--output", required=True, help="Directory to save index files")
    args = p.parse_args()

    chunk_and_embed(args.input, args.output)
