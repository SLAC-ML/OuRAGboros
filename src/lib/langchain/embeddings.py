import os.path

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

import lib.config as config

import lib.langchain.llm as langchain_llms
import lib.langchain.util as langchain_utils

_in_memory_vector_stores = {}


def get_in_memory_vector_store(embedding_model: str) -> VectorStore:
    """
    Returns a basic in-memory vector store for performing a vector search. Useful for
    development.

    :param embedding_model:
    :return:
    """
    if embedding_model not in _in_memory_vector_stores:
        _in_memory_vector_stores[embedding_model] = InMemoryVectorStore(
            get_embedding(embedding_model)
        )

    return _in_memory_vector_stores[embedding_model]


def get_available_embeddings() -> list[str]:
    """
    Retrieves available string embedding models.

    :return:
    """
    ollama_models = langchain_llms.get_available_llms()
    huggingface_models = [
        f"huggingface:{config.huggingface_default_embedding_model}",
    ]

    # If we have a local model trained, we prioritize that one.
    #
    finetuned_model_path = os.path.join(
        config.huggingface_model_cache_folder,
        config.huggingface_finetuned_embedding_model,
    )

    if os.path.exists(finetuned_model_path):
        huggingface_models.insert(0, f"huggingface:{finetuned_model_path}")

    return [*huggingface_models, *ollama_models]


def get_embedding(embedding_model: str) -> Embeddings:
    """
    Returns an embedding for the provided model.

    :param embedding_model:
    :return:
    """
    model_source, model_name = langchain_utils.parse_model_name(embedding_model)

    # Create embedding from model name.
    #
    return (
        HuggingFaceEmbeddings(
            model_name=model_name, cache_folder=config.huggingface_model_cache_folder,
        )
        if model_source == "huggingface"
        else OllamaEmbeddings(model=model_name, base_url=config.ollama_base_url)
    )
