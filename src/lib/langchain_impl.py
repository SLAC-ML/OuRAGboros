from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

from ollama import Client

from lib.config import default_model, default_prompt, opensearch_url, \
    huggingface_embeddings_cache_folder


def _parse_model_name(embedding_model: str):
    """
    Splits the model name in format '<source name>:<source model>' at the first colon and
    returns a tuple of the shape (<source_name>, <source_model>).
    :param embedding_model:
    :return:
    """
    return embedding_model.split(':', maxsplit=1)


def get_llm_models():
    ollama_client = Client()
    return [
        f'ollama:{remote_model['model']}'
        for remote_model in ollama_client.list()['models']
    ] or [f'ollama:{default_model}']


def get_embedding_models():
    ollama_models = get_llm_models()
    huggingface_models = ['huggingface:thellert/physbert_cased']
    return [*huggingface_models, *ollama_models]


def get_embedding(embedding_model: str = default_model) -> Embeddings:
    model_source, model_name = _parse_model_name(embedding_model)

    # Create embedding from model name.
    #
    return (
        HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=huggingface_embeddings_cache_folder
        )
        if model_source == 'huggingface' else
        OllamaEmbeddings(model=model_name)
    )


def pull_model(embedding_model: str = default_model):
    """
    Pulls a specific LLM model.
    :param embedding_model:
    :return:
    """
    model_source, model_name = _parse_model_name(embedding_model)

    if model_source == 'ollama':
        ollama_client = Client()
        ollama_client.pull(model_name)
    elif model_source == 'huggingface':
        # Instantiating the embeddings object forces the model to download.
        #
        get_embedding(embedding_model)


def vectorize_md_docs(
        root_path: str,
        embedding_model: str = default_model,
) -> VectorStore:
    """
    Uses a specific embedding model to perform vector embedding for all Markdown documents
    found via recursive search in the root_path parameter.

    :param root_path:
    :param embedding_model: Expected to follow the format <model_source>:<model_name>
                            (e.g., ollama:deepseek-r1:latest).
    :return:
    """
    embedding = get_embedding(embedding_model)

    loader = DirectoryLoader(
        root_path,
        glob='**/*.md',
        loader_cls=TextLoader,
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
        llm_model: str = default_model,
):
    """
    Asks a particular LLM a question.

    :param question:
    :param context:
    :param llm_model:
    :return:
    """
    model_source, model_name = _parse_model_name(llm_model)

    if model_source == 'ollama':
        ollama_llm = OllamaLLM(model=model_name)
        return ollama_llm.stream([
            SystemMessage(content=default_prompt.format(context)),
            HumanMessage(content=question),
        ])
    else:
        return f'Unsupported LLM source {model_source}'
