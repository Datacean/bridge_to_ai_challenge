"""RAG chatbot for Alice in Wonderland using LangChain + OpenAI.

This script will:
- Load the text of Alice_in_Wonderland.txt
- Split it into chunks with overlap
- Create (or reuse) a Chroma vectorstore
- Use LangChain RetrievalQA with an OpenAI chat model

Usage:
1. Fill `.env.local` with your OPENAI_API_KEY and optional MODEL.
2. Install requirements from `requirements.txt`.
3. Run: python rag_chatbot.py

This is a minimal, educational example. For production use persist the
vectorstore to disk, add proper error handling, and secure your API key.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


def build_or_load_vectorstore(text_path: str, persist_directory: str = None):
	"""Load book, split into chunks, create embeddings and vectorstore.

	If persist_directory is provided and exists, reuse it. Otherwise create
	an in-memory Chroma instance.
	"""
	loader = TextLoader(text_path, encoding="utf-8")
	docs = loader.load()

	splitter = RecursiveCharacterTextSplitter(
		chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", " "]
	)
	docs = splitter.split_documents(docs)

	embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

	if persist_directory:
		vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
		# Persist to disk
		vectordb.persist()
	else:
		vectordb = Chroma.from_documents(documents=docs, embedding=embeddings)

	return vectordb


def build_qa_chain(vectordb):
	model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL, temperature=0.0)
	retriever = vectordb.as_retriever(search_kwargs={"k": 4})
	qa = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=retriever)
	return qa


def interactive_chat(qa):
	print("Alice-in-Wonderland RAG chatbot. Type 'exit' or 'quit' to stop.")
	while True:
		query = input("You: ")
		if not query or query.strip().lower() in {"exit", "quit"}:
			print("Goodbye!")
			break
		resp = qa.run(query)
		print("Assistant:")
		print(resp)


def main():
	repo = Path(__file__).parent
	text_path = repo / "Alice_in_Wonderland.txt"
	if not text_path.exists():
		raise FileNotFoundError(f"Book not found at {text_path}")

	# For small demo projects we persist to ./chroma_persist
	persist_dir = repo / "chroma_persist"
	if persist_dir.exists():
		print("Loading existing vectorstore from disk (chroma_persist)")
		embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
		vectordb = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
	else:
		print("Creating vectorstore from Alice_in_Wonderland.txt (this may take a moment)")
		vectordb = build_or_load_vectorstore(str(text_path), persist_directory=str(persist_dir))

	qa = build_qa_chain(vectordb)
	interactive_chat(qa)


if __name__ == "__main__":
	main()

