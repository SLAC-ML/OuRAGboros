import hashlib
import logging
from typing import Optional, Tuple

import opensearchpy.exceptions
from opensearchpy import OpenSearch

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

from ollama import Client

import lib.config as config

_in_memory_vector_stores = {}
_opensearch_configs = {}


def get_in_memory_vector_store(embedding_model: str) -> VectorStore:
    if embedding_model not in _in_memory_vector_stores:
        _in_memory_vector_stores[embedding_model] = InMemoryVectorStore(
            get_embedding(embedding_model)
        )

    return _in_memory_vector_stores[embedding_model]


def _parse_model_name(embedding_model: str):
    """
    Splits the model name in format '<source name>:<source model>' at the first colon and
    returns a tuple of the shape (<source_name>, <source_model>).
    :param embedding_model:
    :return:
    """
    return embedding_model.split(':', maxsplit=1)


def get_available_llms():
    ollama_client = Client(host=config.ollama_base_url)
    return [
        f'ollama:{remote_model['model']}'
        for remote_model in ollama_client.list()['models']
    ] or [config.default_model]


def get_available_embeddings():
    ollama_models = get_available_llms()
    huggingface_models = [f'huggingface:{config.huggingface_default_embedding_model}']
    return [*huggingface_models, *ollama_models]


def get_embedding(embedding_model: str) -> Embeddings:
    model_source, model_name = _parse_model_name(embedding_model)

    # Create embedding from model name.
    #
    return (
        HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=config.huggingface_model_cache_folder
        )
        if model_source == 'huggingface' else
        OllamaEmbeddings(model=model_name, base_url=config.ollama_base_url)
    )


def pull_model(embedding_model):
    """
    Pulls a specific LLM model.
    :param embedding_model:
    :return:
    """
    model_source, model_name = _parse_model_name(embedding_model)

    if model_source == 'ollama':
        ollama_client = Client(host=config.ollama_base_url)
        ollama_client.pull(model_name)
    elif model_source == 'huggingface':
        # Instantiating the embeddings object forces the model to download.
        #
        get_embedding(embedding_model)


def ask_llm(
        llm_model: str,
        llm_prompt: str,
        question: str,
        context: Optional[str] = None,
):
    """
    Asks a particular LLM a question.

    :param question:
    :param context:
    :param llm_model:
    :param llm_prompt:
    :return:
    """
    model_source, model_name = _parse_model_name(llm_model)

    if model_source == 'ollama':
        ollama_llm = OllamaLLM(model=model_name, base_url=config.ollama_base_url)
        system_message = f"{llm_prompt} \n Context: {context}" if context else llm_prompt

        return ollama_llm.stream([
            SystemMessage(content=system_message),
            HumanMessage(content=question),
        ])
    else:
        return f'Unsupported LLM source {model_source}'


def get_opensearch_index_settings(embedding_model: str) -> Tuple[str, int]:
    if embedding_model not in _opensearch_configs:
        logging.debug(f'{embedding_model} not found in _opensearch_configs; generating..')
        vector_size = len(get_embedding(embedding_model).embed_query('hi'))
        model_id = hashlib.sha256(embedding_model.encode('utf-8')).hexdigest()

        _opensearch_configs[embedding_model] = (
            f'{config.opensearch_index_prefix}_{vector_size}_{model_id}',
            vector_size,
        )

    return _opensearch_configs[embedding_model]


def ensure_opensearch_index(embedding_model_name):
    try:
        index_name, vector_size = get_opensearch_index_settings(
            embedding_model_name)

        opensearch_client = OpenSearch([
            config.opensearch_base_url
        ])
        opensearch_client.indices.create(
            index=index_name,
            body=config.opensearch_index_settings(vector_size=vector_size),
        )
    except opensearchpy.exceptions.RequestError as e:
        if e.status_code != 400:
            raise e


def opensearch_doc_vector_store(embedding_model: str) -> OpenSearchVectorSearch:
    """
    Uses a specific embedding model to perform vector embedding for documents via
    OpenSearch.

    :param embedding_model: Expected to follow the format <model_source>:<model_name>
                            (e.g., ollama:deepseek-r1:latest).
    :param embedding_model:
    :return:
    """
    embeddings = get_embedding(embedding_model)
    index_name, _ = get_opensearch_index_settings(embedding_model)
    return OpenSearchVectorSearch(
        index_name=index_name,
        opensearch_url=config.opensearch_base_url,
        embedding_function=embeddings,
    )
