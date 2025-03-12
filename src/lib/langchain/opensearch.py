import hashlib
import logging
from typing import Tuple

import opensearchpy.exceptions
from opensearchpy import OpenSearch

from langchain_community.vectorstores import OpenSearchVectorSearch

import lib.config as config


import lib.langchain.embeddings as langchain_embeddings

_opensearch_configs = {}

def get_opensearch_index_settings(embedding_model: str) -> Tuple[str, int]:
    """
    Populates an OpenSearch configuration template with a particular model name and vector
    embedding size.

    :param embedding_model:
    :return:
    """
    if embedding_model not in _opensearch_configs:
        logging.debug(f'{embedding_model} not found in _opensearch_configs; generating..')
        vector_size = len(
            langchain_embeddings.get_embedding(embedding_model).embed_query('hi')
        )
        model_id = hashlib.sha256(embedding_model.encode('utf-8')).hexdigest()

        _opensearch_configs[embedding_model] = (
            f'{config.opensearch_index_prefix}_{vector_size}_{model_id}',
            vector_size,
        )

    return _opensearch_configs[embedding_model]


def ensure_opensearch_index(embedding_model_name: str) -> None:
    """
    Ensures the existence of an OpenSearch index. We create one index per embedding model,
    since different models may generate various embedding vector sizes (and even if two
    models generate vectors of the same size, they certainly do not do so the same way).

    :param embedding_model_name:
    :return:
    """
    try:
        index_name, vector_size = get_opensearch_index_settings(embedding_model_name)

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
    :return:
    """
    embeddings = langchain_embeddings.get_embedding(embedding_model)
    index_name, _ = get_opensearch_index_settings(embedding_model)
    return OpenSearchVectorSearch(
        index_name=index_name,
        opensearch_url=config.opensearch_base_url,
        embedding_function=embeddings,
    )
