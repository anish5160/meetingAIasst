"""
Simple Streamlit app that ties everything together.
Run with: streamlit run app_streamlit.py
"""

import streamlit as st
from pathlib import Path
import json

# Import functions from your own scripts
from transcribe import transcribe   # <-- make sure transcribe.py has a function `transcribe(input_path, output_path)`
from chunk_and_embed import chunk_and_embed  # <-- same, define a function in chunk_and_embed.py
from rag_chain import MeetingRAG

st.set_page_config(page_title='Meeting Assistant', layout='wide')
st.title('Meeting Assistant - Demo')

DATA_DIR = Path('data')
INDEX_DIR = DATA_DIR / 'index'
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

uploaded = st.file_uploader('Upload meeting audio (mp3/wav)', type=['mp3','wav'])
if uploaded is not None:
    audio_path = DATA_DIR / 'uploaded_meeting.mp3'
    with open(audio_path, 'wb') as f:
        f.write(uploaded.getbuffer())
    st.audio(str(audio_path))

    st.info('Transcribing...')
    try:
        transcribe(str(audio_path), str(DATA_DIR/'transcript.json'))
        st.success('Transcription finished')
    except Exception as e:
        st.error(f'Error running transcription: {e}')

if (DATA_DIR/'transcript.json').exists():
    if st.button('Create embeddings & index'):
        st.info('Creating embeddings...')
        try:
            chunk_and_embed(str(DATA_DIR/'transcript.json'), str(INDEX_DIR))
            st.success('Index created')
        except Exception as e:
            st.error(f'Error creating index: {e}')

    st.markdown('---')
    st.header('Summarize / Ask questions')
    question = st.text_input('Ask a question about the meeting (or leave blank to get summary)')
    if st.button('Run RAG & Generate'):
        try:
            rag = MeetingRAG(index_dir=str(INDEX_DIR))
            if question.strip()=='' :
                summary = rag.summarize()
                st.subheader('Summary')
                st.write(summary)
                st.subheader('Action Items (raw)')
                st.write(rag.extract_actions())
            else:
                ans = rag.answer(question)
                st.subheader('Answer')
                st.write(ans)
        except Exception as e:
            st.error(f'Error running RAG: {e}')
