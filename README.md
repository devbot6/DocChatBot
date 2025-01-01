# Document Loading Chatbot

This is a chatbot application built using LangChain, Gradio, and HuggingFace embeddings that helps users ask questions related to the loaded document. The chatbot leverages a document-based retrieval system to provide relevant answers based on the content of a PDF book.

## Features

- **PDF Document Loading**: Loads a PDF book and processes its content.
- **Semantic Chunking**: Splits the text into semantically meaningful chunks for better context retrieval.
- **Embeddings and Vector Search**: Uses HuggingFace embeddings and FAISS to create a vector store for efficient similarity-based search.
- **Language Model**: Uses Ollama's Mistral model to generate answers based on the retrieved context.
- **Gradio Interface**: A user-friendly web interface to interact with the chatbot.

## Requirements

Before running the project, ensure you have the following dependencies installed:

- Python 3.7+
- [LangChain](https://www.langchain.com/)
- [Gradio](https://gradio.app/)
- [HuggingFace Embeddings](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com/)

