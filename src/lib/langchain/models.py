import ollama

import lib.config as config

import lib.langchain.embeddings as langchain_embeddings
import lib.langchain.util as langchain_utils

def pull_model(embedding_model: str) -> None:
    """
    Pulls a specific LLM model.
    :param embedding_model:
    :return:
    """
    model_source, model_name = langchain_utils.parse_model_name(embedding_model)

    if model_source == 'ollama':
        ollama.Client(host=config.ollama_base_url).pull(model_name)
    elif model_source == 'huggingface':
        # Instantiating the embeddings object forces the model to download.
        #
        langchain_embeddings.get_embedding(embedding_model)

