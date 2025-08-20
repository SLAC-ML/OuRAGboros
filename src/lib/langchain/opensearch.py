import hashlib
import logging
from typing import Tuple

import opensearchpy.exceptions
from opensearchpy import OpenSearch

from langchain_community.vectorstores import OpenSearchVectorSearch

import lib.config as config


import lib.langchain.embeddings as langchain_embeddings

_opensearch_configs = {}


def get_opensearch_index_settings(
    embedding_model: str, knowledge_base: str = "default"
) -> Tuple[str, int]:
    """
    Populates an OpenSearch configuration template with a particular model name, knowledge base,
    and vector embedding size.

    :param embedding_model: The embedding model name
    :param knowledge_base: The knowledge base name (defaults to "default")
    :return: Tuple of (index_name, vector_size)
    """
    cache_key = f"{embedding_model}#{knowledge_base}"
    if cache_key not in _opensearch_configs:
        logging.debug(f"{cache_key} not found in _opensearch_configs; generating..")
        vector_size = len(
            langchain_embeddings.get_embedding(embedding_model).embed_query("hi")
        )
        model_id = hashlib.sha256(embedding_model.encode("utf-8")).hexdigest()

        # Sanitize knowledge base name for use in index name
        kb_safe = knowledge_base.lower().replace(" ", "_").replace("-", "_")
        kb_safe = "".join(c for c in kb_safe if c.isalnum() or c == "_")

        # For backward compatibility, use original format for "default" knowledge base
        if knowledge_base == "default":
            index_name = f"{config.opensearch_index_prefix}_{vector_size}_{model_id}"
        else:
            index_name = (
                f"{config.opensearch_index_prefix}_{kb_safe}_{vector_size}_{model_id}"
            )

        _opensearch_configs[cache_key] = (index_name, vector_size)

    return _opensearch_configs[cache_key]


def ensure_opensearch_index(
    embedding_model_name: str, knowledge_base: str = "default"
) -> None:
    """
    Ensures the existence of an OpenSearch index. We create one index per embedding model
    and knowledge base combination, since different models may generate various embedding 
    vector sizes (and even if two models generate vectors of the same size, they certainly 
    do not do so the same way).

    :param embedding_model_name: The embedding model name
    :param knowledge_base: The knowledge base name (defaults to "default")
    :return:
    """
    try:
        index_name, vector_size = get_opensearch_index_settings(
            embedding_model_name, knowledge_base
        )

        opensearch_client = OpenSearch([config.opensearch_base_url])
        opensearch_client.indices.create(
            index=index_name,
            body=config.opensearch_index_settings(vector_size=vector_size),
        )
    except opensearchpy.exceptions.RequestError as e:
        if e.status_code != 400:
            raise e


def opensearch_doc_vector_store(
    embedding_model: str, knowledge_base: str = "default"
) -> OpenSearchVectorSearch:
    """
    Uses a specific embedding model to perform vector embedding for documents via
    OpenSearch in a specific knowledge base.

    :param embedding_model: Expected to follow the format <model_source>:<model_name>
                                 (e.g., ollama:deepseek-r1:latest).
    :param knowledge_base: The knowledge base name (defaults to "default")
    :return:
    """
    embeddings = langchain_embeddings.get_embedding(embedding_model)
    index_name, _ = get_opensearch_index_settings(embedding_model, knowledge_base)
    return OpenSearchVectorSearch(
        index_name=index_name,
        opensearch_url=config.opensearch_base_url,
        embedding_function=embeddings,
    )


def get_available_knowledge_bases() -> list[str]:
    """
    Returns a list of available knowledge bases by examining existing OpenSearch indices.
    
    :return: List of knowledge base names
    """
    try:
        opensearch_client = OpenSearch([config.opensearch_base_url])
        indices = opensearch_client.indices.get_alias(
            index=f"{config.opensearch_index_prefix}_*"
        )

        knowledge_bases = set()
        prefix = config.opensearch_index_prefix + "_"

        logging.debug(f"Found indices: {list(indices.keys())}")

        for index_name in indices.keys():
            if index_name.startswith(prefix):
                # Parse index name to extract knowledge base
                # Format examples:
                # - ouragboros_768_abc123... (old format = "default")
                # - ouragboros_test_kb_768_abc123... (new format = "test_kb")

                remainder = index_name[len(prefix) :]
                parts = remainder.split("_")

                logging.debug(f"Parsing index: {index_name}, parts: {parts}")

                if len(parts) >= 2:
                    # Look for the vector size (should be a number like 768)
                    # Vector sizes are typically 64-8192, so 2-4 digits
                    vector_size_index = None
                    for i, part in enumerate(parts):
                        if (
                            part.isdigit() and 2 <= len(part) <= 4
                        ):  # Vector sizes are 64-8192
                            try:
                                size = int(part)
                                if 32 <= size <= 8192:  # Reasonable vector size range
                                    vector_size_index = i
                                    break
                            except ValueError:
                                continue

                    if vector_size_index is not None:
                        if vector_size_index == 0:
                            # Format: ouragboros_768_hash... (old format)
                            knowledge_bases.add("default")
                        else:
                            # Format: ouragboros_kb_name_768_hash... (new format)
                            kb_name = "_".join(parts[:vector_size_index])
                            knowledge_bases.add(kb_name)
                            logging.debug(f"Found knowledge base: {kb_name}")
                    else:
                        # Fallback: assume it's default if we can't parse
                        knowledge_bases.add("default")

        # Always include "default" if we have any indices
        if indices:
            knowledge_bases.add("default")

        kb_list = sorted(list(knowledge_bases))
        logging.debug(f"Final knowledge bases list: {kb_list}")
        return kb_list if kb_list else ["default"]

    except Exception as e:
        logging.warning(f"Failed to fetch knowledge bases: {e}")
        return ["default"]


def delete_knowledge_base(knowledge_base: str, embedding_model: str) -> bool:
    """
    Deletes a knowledge base by removing its OpenSearch index.
    
    :param knowledge_base: The knowledge base name to delete
    :param embedding_model: The embedding model used (needed to find correct index)
    :return: True if successful, False otherwise
    """
    if knowledge_base == "default":
        raise ValueError("Cannot delete the default knowledge base")

    try:
        index_name, _ = get_opensearch_index_settings(embedding_model, knowledge_base)
        opensearch_client = OpenSearch([config.opensearch_base_url])

        # Check if index exists
        if opensearch_client.indices.exists(index=index_name):
            opensearch_client.indices.delete(index=index_name)
            logging.info(
                f"Successfully deleted knowledge base '{knowledge_base}' (index: {index_name})"
            )
            return True
        else:
            logging.warning(
                f"Knowledge base '{knowledge_base}' does not exist (index: {index_name})"
            )
            return False

    except Exception as e:
        logging.error(f"Failed to delete knowledge base '{knowledge_base}': {e}")
        return False
