from typing import Optional, Iterator

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaLLM
from langchain_openai import OpenAI

import ollama
import openai

import lib.config as config

import lib.langchain.util as langchain_utils

_ollama_client = lambda: ollama.Client(host=config.ollama_base_url)
_openai_client = lambda: openai.Client(api_key=config.openai_api_key)


def get_available_llms() -> list[str]:
    """
    This function is responsible for generating the LLM names available for use. The
    general format should be: "<llm provider>:<llm model name>". For example:

    "ollama:deepseek-r1:latest"
    "openai:o1-mini"

    :return:
    """
    ollama_models = _ollama_client().list()['models']
    openai_models = _openai_client().models.list() if config.openai_api_key else []

    return [
        *[f'ollama:{remote_model['model']}' for remote_model in ollama_models],
        *[f'openai:{remote_model.id}' for remote_model in openai_models],
    ] or [config.default_model]


def query_llm(
        llm_model: str,
        llm_prompt: str,
        question: str,
        context: Optional[str] = None,
) -> Iterator[str]:
    """
    Asks a particular LLM a question.

    :param question:
    :param context:
    :param llm_model:
    :param llm_prompt:
    :return:
    """
    model_source, model_name = langchain_utils.parse_model_name(llm_model)

    if model_source == 'ollama':
        ollama_llm = OllamaLLM(model=model_name, base_url=config.ollama_base_url)
        system_message = f"{llm_prompt} \n Context: {context}" if context else llm_prompt

        return ollama_llm.stream([
            SystemMessage(content=system_message),
            HumanMessage(content=question),
        ])
    if model_source == 'openai':
        openai_llm = OpenAI(openai_api_key=config.openai_api_key, model_name=model_name)
        system_message = f"{llm_prompt} \n Context: {context}" if context else llm_prompt

        return openai_llm.stream([
            SystemMessage(content=system_message),
            HumanMessage(content=question),
        ])
    else:
        yield f'Unsupported LLM source {model_source}'
