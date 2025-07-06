
# 💡 Chat with Your Notes

A modern, interactive Streamlit application that allows you to **chat with your PDF notes** using **local LLMs (like Ollama)** and **semantic search with FAISS**. Whether you're studying, summarizing content, or just revisiting material, this tool helps you understand your documents quickly and effectively.

---

## 📌 Introduction

**Chat with Your Notes** is a privacy-friendly PDF question-answering tool that allows users to upload a PDF, process its content locally, and ask questions about it. It uses:

- **Sentence-BERT embeddings** to understand the meaning of your content.
- **FAISS** for fast and efficient similarity search.
- **Ollama (local LLM)** to generate accurate, context-based answers.

All of this runs **completely offline** (no API keys needed!) with your own **local LLM** like `phi3` served by Ollama.

---

## 🚀 Features

- 📄 Upload any PDF file
- 🧠 Split and embed content semantically
- 🔍 Ask any question from your notes
- 🗂️ Get answers grounded in actual context
- 💅 Beautiful modern UI (fully responsive & styled)

---

## 🧰 Tech Stack

- **Frontend:** Streamlit
- **PDF Processing:** PyPDF2
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Store:** FAISS
- **LLM Backend:** Ollama (e.g. `phi3`, `mistral`, etc.)
- **Styling:** Custom CSS injected into Streamlit

---

## ⚙️ How It Works

1. **Upload PDF:** Extracts and splits text into sentences and semantic chunks.
2. **Generate Embeddings:** Each chunk is converted into a vector using Sentence-BERT.
3. **Build FAISS Index:** Chunks are stored for fast nearest-neighbor search.
4. **Ask Question:** User asks a question; the app finds the most relevant chunk.
5. **LLM Response:** The chunk and question are sent to a local Ollama LLM to generate the answer.

---

## 🧪 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/chat-with-your-notes.git
cd chat-with-your-notes
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure you have:
- Python 3.8+
- Ollama installed and running: [https://ollama.com](https://ollama.com)

### 3. Run Ollama Model

```bash
ollama run phi3
```

> You can change the model to `mistral`, `llama3`, etc., in the code (`model="phi3"`).

### 4. Launch the App

```bash
streamlit run app.py
```

---

## 📸 UI Preview

> Screenshot: Upload PDF, ask questions, get contextual answers.

![UI Preview](preview.png)

---

## 📄 Example PDF Use Cases

- Study class notes
- Read and question research papers
- Chat with technical documentation
- Summarize books or chapters

---

## 📚 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Ollama](https://ollama.com)
- [SentenceTransformers](https://www.sbert.net/)
- [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)

---

## ❤️ Made with love

> Powered by Streamlit · Embeddings by SBERT · LLM by Ollama  
> 100% private. 100% offline. 100% helpful.
