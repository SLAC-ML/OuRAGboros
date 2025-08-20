def parse_model_name(embedding_model: str):
    """
    Splits the model name in format '<source name>:<source model>' at the first colon and
    returns a tuple of the shape (<source_name>, <source_model>).

    This function is primarily consumed by the lib.langchain model for sending various
    LLM/embedding operations to their corresponding implementations.

    :param embedding_model:
    :return:
    """
    return embedding_model.split(":", maxsplit=1)
