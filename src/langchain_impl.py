from typing import Optional

from langchain_core.vectorstores import VectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import (
    OpenSearchVectorSearch
)
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from ollama import Client

from config import default_model, default_prompt, opensearch_url


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
) -> VectorStore:
    """
    Uses a specific Ollama model to perform vector embedding for all Markdown documents
    found via recursive search in the root_path parameter.

    :param root_path:
    :param ollama_model:
    :return:
    """
    embedding = OllamaEmbeddings(model=ollama_model)
    loader = DirectoryLoader(
        root_path,
        glob='**/*.md',
        loader_cls=UnstructuredMarkdownLoader,
    )
    documents = loader.load()

    from langchain_core.vectorstores import InMemoryVectorStore
    return InMemoryVectorStore.from_documents(documents, embedding)
    # return OpenSearchVectorSearch.from_documents(
    #     documents,
    #     embedding,
    #     opensearch_url=opensearch_url,
    # )


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
    return ollama_llm.stream([
        SystemMessage(content=default_prompt.format(context)),
        HumanMessage(content=question),
    ])
