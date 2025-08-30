from typing import Iterator, AsyncGenerator
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

# Global async client for Stanford API (connection pooling + async)
_stanford_async_client = None

def _get_stanford_async_client():
    """Get or create shared async Stanford client for connection pooling"""
    global _stanford_async_client
    if _stanford_async_client is None and config.stanford_api_key:
        _stanford_async_client = openai.AsyncOpenAI(
            api_key=config.stanford_api_key,
            base_url=config.stanford_base_url
        )
    return _stanford_async_client

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
    
    # Mock LLM for performance testing - returns instant response
    if config.use_mock_llm:
        def mock_llm_stream():
            mock_response = f"Mock response to: {question[:50]}{'...' if len(question) > 50 else ''}"
            # Split response into tokens to simulate streaming
            words = mock_response.split()
            for word in words:
                yield word + " "
        
        print(f"🤖 MOCK LLM: Using mock response for {llm_model}")
        return mock_llm_stream()
    
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

        # Wrap Stanford API streaming with validation error handling
        def safe_stanford_stream():
            tokens_yielded = []
            try:
                stream = stanford_llm.stream([
                    SystemMessage(content=system_message),
                    HumanMessage(content=question)
                ])
                for token in stream:
                    # Filter out invalid tokens that cause validation errors
                    if token is not None:
                        # Convert any token format to string safely
                        token_str = None
                        if hasattr(token, 'content') and token.content is not None:
                            token_str = str(token.content)
                        elif hasattr(token, 'text') and token.text is not None:
                            token_str = str(token.text) 
                        elif isinstance(token, str):
                            token_str = token
                        
                        if token_str:
                            tokens_yielded.append(token_str)
                            yield token_str
                            
            except Exception as e:
                # If Stanford streaming fails partway through, we've already yielded some tokens
                # Don't raise the error if we got valid tokens - this prevents fallback mode
                if tokens_yielded:
                    print(f"Stanford streaming completed with validation error at end: {e}")
                    print(f"Successfully yielded {len(tokens_yielded)} tokens, ignoring final validation error")
                    return  # Exit gracefully without raising error
                else:
                    # If no tokens were yielded, there's a real problem
                    print(f"Stanford streaming failed completely: {e}")
                    raise e
                
        return safe_stanford_stream()
    elif model_source == 'google':
        google_llm = ChatGoogleGenerativeAI(google_api_key=config.google_api_key, model=model_name, max_tokens=max_tokens)

        return google_llm.stream([
            SystemMessage(content=system_message),
            HumanMessage(content=question)
        ])
    else:
        return iter(())


async def query_llm_async(
        llm_model: str,
        question: str,
        system_message: str = "",
        max_tokens: int = 2000,
) -> AsyncGenerator[str, None]:
    """
    Async version of query_llm for better concurrency performance.
    Uses native async OpenAI client to avoid blocking the event loop.
    
    :param question: The user's question
    :param system_message: System prompt
    :param llm_model: Model specification (e.g., "stanford:gpt-4o-mini") 
    :param max_tokens: Maximum tokens to generate
    :return: Async generator yielding token strings
    """
    
    # Mock LLM for performance testing - returns instant response
    if config.use_mock_llm:
        mock_response = f"Mock response to: {question[:50]}{'...' if len(question) > 50 else ''}"
        # Split response into tokens to simulate streaming
        words = mock_response.split()
        for word in words:
            yield word + " "
        return
    
    model_source, model_name = langchain_utils.parse_model_name(llm_model)
    
    if model_source == 'stanford':
        # Use native async OpenAI client for Stanford API
        async_client = _get_stanford_async_client()
        if not async_client:
            raise Exception("Stanford API key not configured")
        
        try:
            # Native async streaming call - no blocking!
            stream = await async_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question}
                ],
                stream=True,
                max_tokens=max_tokens
            )
            
            # Async iteration over stream - truly concurrent!
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            
        except Exception as e:
            # Log error but re-raise for fallback handling
            print(f"Stanford async streaming failed: {e}")
            raise e
    
    else:
        # For other models, fall back to sync approach (wrapped in async)
        # TODO: Implement async versions for other providers
        sync_generator = query_llm(llm_model, question, system_message, max_tokens)
        
        # Convert sync generator to async (not ideal but maintains compatibility)
        import asyncio
        loop = asyncio.get_event_loop()
        
        def get_next_token():
            try:
                return next(sync_generator)
            except StopIteration:
                return None
        
        while True:
            token = await loop.run_in_executor(None, get_next_token)
            if token is None:
                break
            yield token
