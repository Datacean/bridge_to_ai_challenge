# Bridge To AI Challenge - Document (RAG) Chatbot

This folder contains a small retrieval-augmented generation (RAG) chatbot that lets you ask questions about the book "Alice in Wonderland".

Files of interest:
- `rag_chatbot.py` - the main script. It builds or loads a vector store from the book, then starts an interactive QA loop.
- `Alice_in_Wonderland.txt` - the source book used for retrieval.
- `.env.local` - template for environment variables (copy to `.env` and fill your OpenAI API key).
- `requirements.txt` - Python dependencies required to run the example.

Quick start (conda + pip)
-------------------------

1. Create and activate a conda environment (recommended):

```bash
conda create -n btaic python=3.12 -y
conda activate btaic
```

2. Install the Python dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the environment template and set your OpenAI key:

```bash
cp .env.local .env
# Edit 03_document_chatbot/.env and set OPENAI_API_KEY with your key
```

4. Run the RAG chatbot:

```bash
python rag_chatbot.py
```

What to expect
--------------
- On first run the script will split the text, compute embeddings, and persist a vector store (Chroma) in a local folder.
- Subsequent runs will reuse the persisted vector store for faster startup.
- The script uses OpenAI for embeddings and answer generation via LangChain. Set `OPENAI_MODEL` in the environment if you want to override the default.

Notes
-----
- Keep your API key secret. Do not check real keys into git.
- If you want different behavior (chunk sizes, retriever settings, or a different vector store), edit `rag_chatbot.py`.
- Large-scale or production usage requires more robust chunking, metadata, and privacy considerations.

Example queries
---------------
- "Who is the Queen of Hearts?"
- "Summarize the first chapter."
- "Where does Alice first meet the White Rabbit?"


