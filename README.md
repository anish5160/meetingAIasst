# Meeting Assistant - Starter Scaffold

This scaffold gives you a minimal, easy-to-understand Meeting Assistant prototype that:
1. Transcribes audio using OpenAI Whisper (local)
2. Chunks transcript and creates embeddings (sentence-transformers)
3. Indexes embeddings into FAISS
4. Uses a simple RAG flow with OpenAI (or local LLM) via LangChain to produce summaries and extract action items
5. Streamlit app to upload audio and see results

IMPORTANT: This scaffold expects you to run it locally (your laptop). It uses standard Python libraries and common APIs. See "Prerequisites" below.

---

## Files in this folder

- `transcribe.py` : Run whisper on an audio file to get transcript (produces JSON).
- `chunk_and_embed.py` : Chunk transcript and compute embeddings, store FAISS index.
- `rag_chain.py` : LangChain wrapper: retrieval + summarization + action extraction.
- `app_streamlit.py` : Simple Streamlit UI to upload audio, run pipeline and show results.
- `requirements.txt` : Python packages to install.
- `run_demo.sh` : Simple script showing example commands.
- `sample_audio/` : (You should place one `meeting.mp3` here to test.)

---

## Quick Setup (commands to run locally)

1. Create venv and install:
```bash
python -m venv venv
source venv/bin/activate      # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

2. Put a short meeting audio file at `sample_audio/meeting.mp3`.

3. Transcribe:
```bash
python transcribe.py --input meeting.mp3 --output data/transcript.json
```

4. Create embeddings & index:
```bash
python chunk_and_embed.py --input data/transcript.json --output data/index
```

5. Start Streamlit app:
```bash
streamlit run app_streamlit.py
```

Open the Streamlit URL shown (usually http://localhost:8501) and upload the same audio file from the UI. The app will transcribe (or reuse existing transcript), produce a summary and list of action items.

---

## If you don't have GPU or prefer API-based transcription:
- You can use AssemblyAI, Google STT, or OpenAI Whisper API (if available) in `transcribe.py` by setting a flag. See inline comments.

---

## Notes for professor / report
- This scaffold uses **Whisper** for ASR, **sentence-transformers** for embeddings, **FAISS** for local vector store, and **LangChain/OpenAI** for generation.
- The main research claim: combining ASR + RAG + action-item extraction reduces time-to-find and provides evidence-backed minutes.