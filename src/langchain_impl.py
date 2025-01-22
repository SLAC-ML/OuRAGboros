import os
from typing import Optional

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)

from ollama import Client
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Fetch model as needed
#
default_model = os.environ.get('OLLAMA_MODEL_DEFAULT', default='deepseek-r1:latest')

default_prompt = """You are an assistant tasked with helping students get acquainted with 
a new research project designed to make sense of a long series of scientific logs 
written by equipment operators at the Stanford Linear Accelerator. If you don't know 
the answer, say you don't know. Keep answers concise. Encourage students to reach out 
to the listed collaborators.
Context: {}"""

def get_models():
    ollama_client = Client()
    return (
            [remote_model['model'] for remote_model in ollama_client.list()['models']]
            or [default_model]
    )

def pull_model(ollama_model: str = default_model):
    """
    Pulls a specific LLM model.
    :param ollama_model:
    :return:
    """
    ollama_client = Client()
    ollama_client.pull(ollama_model)


def vectorize_md_docs(
        root_path: str,
        ollama_model: str = default_model,
) -> InMemoryVectorStore:
    """
    Uses a specific Ollama model to perform vector embedding for all Markdown documents
    found via recursive search in the root_path parameter.

    :param root_path:
    :param ollama_model:
    :return:
    """
    embedding = OllamaEmbeddings(model=ollama_model)
    loader = DirectoryLoader(
        root_path, glob='**/*.md', loader_cls=UnstructuredMarkdownLoader
    )
    documents = loader.load()

    return InMemoryVectorStore.from_documents(documents, embedding)


def ask_llm(
        question: str,
        context: Optional[str] = None,
        ollama_model: str = default_model,
):
    """
    Asks a particular LLM a question.

    :param question:
    :param context:
    :param ollama_model:
    :return:
    """
    ollama_llm = OllamaLLM(model=ollama_model)
    return ollama_llm.invoke([
        SystemMessage(content=default_prompt.format(context)),
        HumanMessage(content=question),
    ])
