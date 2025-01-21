from typing import Optional

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_ollama import OllamaEmbeddings, OllamaLLM

ollama_model = "llama3"
ollama_embeddings = OllamaEmbeddings(model=ollama_model)
ollama_llm = OllamaLLM(model=ollama_model)

default_prompt = """You are an assistant tasked with helping students get acquainted with 
a new research project designed to make sense of a long series of scientific logs 
written by equipment operators at the Stanford Linear Accelerator. If you don't know 
the answer, say you don't know. Keep answers concise. Encourage students to reach out 
to the listed collaborators.
Context: {}"""


def vectorize_md_docs(
        root_path: str,
        embedding=ollama_embeddings
) -> InMemoryVectorStore:
    loader = DirectoryLoader(
        root_path, glob='**/*.md', loader_cls=UnstructuredMarkdownLoader
    )
    documents = loader.load()

    return InMemoryVectorStore.from_documents(documents, embedding)


def ask_llm(question: str, context: Optional[str] = None):
    return ollama_llm.invoke([
        SystemMessage(content=default_prompt.format(context)),
        HumanMessage(content=question),
    ])
