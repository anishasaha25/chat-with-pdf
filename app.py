import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import re

# --- Enhanced, Modern Custom CSS for "Chat with Your Notes" ---
st.markdown("""
    <style>
    html, body, .stApp {
        background: linear-gradient(120deg, #232526 0%, #2c3e50 100%);
        color: #f5f6fa;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .main {
        background: transparent;
    }
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
        max-width: 700px;
        margin: auto;
    }
    .stTitle {
        color: #00e676;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-align: center;
        margin-bottom: 2rem;
        margin-top: 0.5rem;
        text-shadow: 0 2px 12px #000a;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .stFileUploader, .stTextInput, .stButton > button {
        background: #23272f !important;
        color: #f5f6fa !important;
        border-radius: 12px !important;
        border: 1.5px solid #00e676 !important;
        font-size: 1.13rem !important;
        margin-bottom: 0.5rem !important;
    }
    .stTextInput > div > div > input {
        background: #23272f !important;
        color: #f5f6fa !important;
        border-radius: 12px !important;
        border: 1.5px solid #00e676 !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #00e676 0%, #00bfae 100%) !important;
        color: #232526 !important;
        font-weight: 700 !important;
        border: none !important;
        margin-top: 0.5rem !important;
        margin-bottom: 1rem !important;
        transition: background 0.2s;
        box-shadow: 0 2px 8px #00e67633;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #00bfae 0%, #00e676 100%) !important;
        color: #fff !important;
    }
    .answer-block {
        background: #23272f;
        border-radius: 14px;
        padding: 1.3rem 1.3rem 0.8rem 1.3rem;
        margin-bottom: 1.7rem;
        border-left: 6px solid #00e676;
        box-shadow: 0 2px 16px #0004;
        font-size: 1.13rem;
    }
    .context-block {
        background: #1a1d22;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-top: 0.8rem;
        color: #b2bec3;
        font-size: 1.04rem;
        border-left: 4px solid #00bfae;
        box-shadow: 0 1px 6px #00bfae22;
    }
    .stMarkdown, .stText {
        color: #f5f6fa !important;
    }
    .stFileUploader label {
        color: #00e676 !important;
        font-weight: 700 !important;
        font-size: 1.13rem !important;
    }
    .stTextInput label {
        color: #00e676 !important;
        font-weight: 700 !important;
        font-size: 1.13rem !important;
    }
    .footer {
        text-align: center;
        color: #b2bec3;
        font-size: 0.98rem;
        margin-top: 2.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }
    </style>
""", unsafe_allow_html=True)

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def split_text_to_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_sentences(sentences, chunk_size=5, overlap=2):
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = " ".join(sentences[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def ask_ollama(question, context, model="phi3", max_tokens=100):
    prompt = f"Answer the question based only on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens}
        }
    )
    if response.ok:
        return response.json()["response"].strip()
    else:
        return "Error: Could not get response from LLM."

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- Modern Heading ---
st.markdown('<div class="stTitle">💡 Chat with Your Notes</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")
if uploaded_file:
    text = get_pdf_text(uploaded_file)
    sentences = split_text_to_sentences(text)
    chunks = chunk_sentences(sentences)
    model = load_model()
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    user_question = st.text_input("💬 Ask a question about your notes:")
    if user_question:
        q_emb = model.encode([user_question])
        D, I = index.search(np.array(q_emb), k=1)  # Only top 1 chunk for speed!
        context = chunks[I[0][0]]
        st.markdown('<div class="answer-block">', unsafe_allow_html=True)
        st.markdown("**Answer:**")
        answer = ask_ollama(user_question, context, model="phi3", max_tokens=100)
        st.markdown(f"{answer}")
        st.markdown('<div class="context-block">', unsafe_allow_html=True)
        st.markdown("**Context used:**")
        st.write(context)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- Footer for a polished look ---
st.markdown('<div class="footer">Made with ❤️ for your notes · Powered by Streamlit & Ollama</div>', unsafe_allow_html=True)
