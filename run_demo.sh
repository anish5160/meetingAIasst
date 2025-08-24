#!/bin/bash
python transcribe.py --input sample_audio/meeting.mp3 --output data/transcript.json
python chunk_and_embed.py --input data/transcript.json --output data/index
streamlit run app_streamlit.py