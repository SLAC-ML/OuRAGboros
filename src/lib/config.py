"""
This file contains all dynamic configuration for the application (API keys, server
locations, etc.).
"""
import os

import dotenv

dotenv.load_dotenv('.default.env')


class Env(str):
    """
    Contains all environment variable names used in this application.
    """
    OLLAMA_BASE_URL = 'OLLAMA_BASE_URL'
    OLLAMA_MODEL_DEFAULT = 'OLLAMA_MODEL_DEFAULT'
    OPENAI_API_KEY = 'OPENAI_API_KEY'
    OPENAI_BASE_URL = 'OPENAI_BASE_URL'
    STANFORD_API_KEY = 'STANFORD_API_KEY'
    STANFORD_BASE_URL = 'STANFORD_BASE_URL'
    GOOGLE_API_KEY = 'GOOGLE_API_KEY'
    SENTENCE_TRANSFORMERS_HOME = 'SENTENCE_TRANSFORMERS_HOME'
    HUGGINGFACE_EMBEDDING_MODEL_DEFAULT = 'HUGGINGFACE_EMBEDDING_MODEL_DEFAULT'
    HUGGINGFACE_FINETUNED_EMBEDDING_MODEL = 'HUGGINGFACE_FINETUNED_EMBEDDING_MODEL'
    PDF_PARSER_MODEL = 'PDF_PARSER_MODEL'
    PREFER_OPENSEARCH = 'PREFER_OPENSEARCH'
    OPENSEARCH_BASE_URL = 'OPENSEARCH_BASE_URL'
    OPENSEARCH_INDEX_PREFIX = 'OPENSEARCH_INDEX_PREFIX'
    QDRANT_BASE_URL = 'QDRANT_BASE_URL'
    PREFER_QDRANT = 'PREFER_QDRANT'
    USE_MOCK_EMBEDDINGS = 'USE_MOCK_EMBEDDINGS'

def _bool_from_env(env: str) -> bool:
    value = os.getenv(env).lower() if os.getenv(env) else None

    true_vals = ('yes', 'true', 't', 'y', '1')
    false_vals = ('no', 'false', 'f', 'n', '0', 'off', None, '')

    if value in true_vals:
        return True
    elif value in false_vals:
        return False
    else:
        raise ValueError(
            f'Boolean value expected for environment variable {env}. Choose from '
            f'{true_vals} or {false_vals}. Value given was ${value}.'
        )


# LLM Configuration
#
ollama_base_url = os.getenv(Env.OLLAMA_BASE_URL)
default_model = os.getenv(Env.OLLAMA_MODEL_DEFAULT)

openai_api_key = os.getenv(Env.OPENAI_API_KEY)
openai_base_url = os.getenv(Env.OPENAI_BASE_URL)
stanford_api_key = os.getenv(Env.STANFORD_API_KEY)
stanford_base_url = os.getenv(Env.STANFORD_BASE_URL)
google_api_key = os.getenv(Env.GOOGLE_API_KEY)

#default_prompt = (
#    "You are a helpful assistant. Output answers in Markdown. Use $ and $$ to surround "
#    "mathematical formulas. Try to tie your answer to the provided list of sources. Say "
#    "you don't know if you can't. Be as concise as possible."
#)
default_prompt = (
"You are a knowledgeable physics researcher explaining complex concepts. Use $ and $$ to surround mathematical formulas."
"Example:"
"1. The speed of light is denoted by $c$."
"2. Einstein's famous equation is given by $E=mc^2$."
"3. This phenomenon occurs when the particle's velocity exceeds the local speed of light, resulting in a characteristic cone of radiation emitted at an angle relative to the particle's direction, described by the Cherenkov angle $\theta_{c}$."
"First, identify the core concept. Second, explain its definition. Third, provide relevant formulas. Fourth, discuss its implications based on the sources. Finally, present the answer."
"After generating the answer, review it to ensure it directly addresses the query, is concise, and uses proper LaTeX formatting. Try to tie your answer to the provided list of sources. Say you don't know if you can't. If the answer is not directly supported by the provided sources, state clearly that the information is not available in the given context. If there is any uncertainty, please indicate the level of confidence in your answer. Be as concise as possible."
)

test_text = r"""
We consider a symmetric matrix A and the Lanczos iteration starting from vector $q_1$. We
assume that we can run all $n-1$ iterations successfully. $Q = [q_1, q_2, ..., q_n]$ is
the orthogonal Lanczos sequences. We denote by T the tri-diagonal matrix such that

$$
A = Q T Q^T.
$$

We also denote by $T_k$ the $k \times k$ leading principal submatrix of $T$ and $Q_k$ the
$n \times k$ matrix formed by the first $k$ columns of $Q$.

Prove that

$$
T_k = \arg \min_{R} {\left\lVert {A Q_k - Q_k R} \right\rVert_2},
$$

where $R$ is a $k \times k$ matrix.
"""

# Local embeddings cache folder. Only used for HuggingFace embeddings.
#
# See https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.html#langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.cache_folder
huggingface_model_cache_folder = os.getenv(Env.SENTENCE_TRANSFORMERS_HOME)
huggingface_default_embedding_model = os.getenv(Env.HUGGINGFACE_EMBEDDING_MODEL_DEFAULT)
huggingface_finetuned_embedding_model = os.getenv(
    Env.HUGGINGFACE_FINETUNED_EMBEDDING_MODEL
)
pdf_parser_model = os.getenv(Env.PDF_PARSER_MODEL)

# Vector store configuration
#
prefer_opensearch = _bool_from_env(Env.PREFER_OPENSEARCH)
opensearch_base_url = os.getenv(Env.OPENSEARCH_BASE_URL)
opensearch_index_prefix = os.getenv(Env.OPENSEARCH_INDEX_PREFIX)

# Qdrant configuration
prefer_qdrant = _bool_from_env(Env.PREFER_QDRANT)
qdrant_base_url = os.getenv(Env.QDRANT_BASE_URL, 'http://localhost:6333')

# Mock embeddings for performance testing
use_mock_embeddings = _bool_from_env(Env.USE_MOCK_EMBEDDINGS)


def opensearch_index_settings(vector_size: int):
    return {
        'settings': {
            'index': {
                'number_of_shards': 1,
                'number_of_replicas': 0,
                'knn': True
            }
        },
        'mappings': {
            'properties': {
                # For keyword search
                #
                'content': {
                    'type': 'text',
                },
                # Vector search
                #
                'vector_field': {
                    'type': 'knn_vector',
                    'dimension': vector_size,
                    'space_type': 'l2',
                    # See https://opensearch.org/docs/latest/search-plugins/knn/knn-index#method-definitions
                    #
                    'method': {
                        'name': 'hnsw',
                        'engine': 'lucene',
                        'parameters': {
                            'ef_construction': 512,
                            'm': 16
                        }
                    }
                },
            }
        }
    }


text_file_types = ['txt', 'md', 'tex']
