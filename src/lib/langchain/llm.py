from typing import Iterator
import logging

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaLLM
from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

import ollama
import openai
from google import genai

import lib.config as config

import lib.langchain.util as langchain_utils

def _ollama_client():
    return ollama.Client(host=config.ollama_base_url)

def _openai_client():
    return openai.Client(api_key=config.openai_api_key, base_url=config.openai_base_url)

def _stanford_client():
    return openai.Client(api_key=config.stanford_api_key, base_url=config.stanford_base_url)

def _google_client():
    return genai.Client(api_key=config.google_api_key)

_logger = logging.Logger(__name__)


def get_available_llms() -> list[str]:
    """
    This function is responsible for generating the LLM names available for use. The
    general format should be: "<llm provider>:<llm model name>". For example:

    "ollama:deepseek-r1:latest"
    "openai:o1-mini"

    :return:
    """
    ollama_models = []
    try:
        ollama_models = _ollama_client().list()['models']
    except Exception:
        _logger.error("Failed to fetch Ollama models. Is the Ollama API accessible?",
                      exc_info=True)

    openai_models = []
    try:
        all_models = _openai_client().models.list() if config.openai_api_key else []
        openai_models = [m for m in all_models if m.id.startswith("gpt-")]
    except Exception:
        _logger.error("Failed to fetch OpenAI models. Is the OpenAI API accessible?",
                      exc_info=True)

    stanford_models = []
    try:
        all_models = _stanford_client().models.list() if config.stanford_api_key else []
        stanford_models = [m for m in all_models]
    except Exception:
        _logger.error("Failed to fetch Stanford AI models. Is the Stanford AI API accessible?",
                      exc_info=True)

    google_models = []
    try:
        google_models = _google_client().models.list() if config.google_api_key else []
    except Exception:
        _logger.error("Failed to fetch Google models. Is the Google API accessible?",
                      exc_info=True)

    return [
        *[f'ollama:{remote_model['model']}' for remote_model in ollama_models],
        *[f'openai:{remote_model.id}' for remote_model in openai_models],
        *[f'stanford:{remote_model.id}' for remote_model in stanford_models],
        *[f'google:{remote_model.name}' for remote_model in google_models],
    ] or [config.default_model]


def query_llm(
        llm_model: str,
        question: str,
        system_message: str = "",
        max_tokens: int = 2000, # token limit
) -> Iterator[str]:
    """
    Asks a particular LLM a question. A system message (e.g., "you are a helpful
    teaching assistant tasked with explaining the following documents to students...")
    may optionally be provided to the LLM.

    :param question:
    :param system_message:
    :param llm_model:
    :param max_tokens: The maximum number of tokens to generate in the response.
    :return:
    """
    model_source, model_name = langchain_utils.parse_model_name(llm_model)
    print("$$$$$$$$$$$$model_source:", model_source)

    if model_source == 'ollama':
        ollama_llm = OllamaLLM(model=model_name, base_url=config.ollama_base_url)
        return ollama_llm.stream([
            SystemMessage(content=system_message),
            HumanMessage(content=question)
        ])
    elif model_source == 'openai':
        openai_llm = OpenAI(
            openai_api_key=config.openai_api_key,
            openai_api_base=config.openai_base_url,
            model_name=model_name, 
            max_tokens=max_tokens
        )

        return openai_llm.stream([
            SystemMessage(content=system_message),
            HumanMessage(content=question)
        ])
    elif model_source == 'stanford':
        stanford_llm = OpenAI(
            openai_api_key=config.stanford_api_key,
            openai_api_base=config.stanford_base_url,
            model_name=model_name, 
            max_tokens=max_tokens
        )

        return stanford_llm.stream([
            SystemMessage(content=system_message),
            HumanMessage(content=question)
        ])
    elif model_source == 'google':
        google_llm = ChatGoogleGenerativeAI(google_api_key=config.google_api_key, model=model_name, max_tokens=max_tokens)

        return google_llm.stream([
            SystemMessage(content=system_message),
            HumanMessage(content=question)
        ])
    else:
        return iter(())
