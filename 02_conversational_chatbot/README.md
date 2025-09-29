# Bridge To AI Challenge - Conversational Chatbot

This folder contains a minimal terminal chatbot that uses the OpenAI Responses API.

Files of interest:
- `simple_chatbot.py` - interactive chatbot you can run from the terminal.
- `.env` - local environment variables (copy and fill `OPENAI_API_KEY`).
- `requirements.txt` - pip dependencies.

Quick start (conda + pip)
-------------------------

1. Create and activate a conda environment:

```bash
conda create -n btaic python=3.12 -y
conda activate btaic
```

2. Install Python dependencies with pip:

```bash
pip install -r requirements.txt
```

3. Copy `.env` and set your OpenAI API key:

```bash
cp .env.local .env
# Edit 02_conversational_chatbot/.env and replace with your real key
```

4. Run the chatbot:

```bash
python simple_chatbot.py
```

Notes
-----
- The script reads `OPENAI_API_KEY` and optional `OPENAI_MODEL` from the environment.
- Keep your API key secret. Do not check real keys into git.

