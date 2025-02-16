import os

import dotenv

dotenv.load_dotenv('.default.env')


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
            f'{true_vals} or {false_vals}.'
        )


# LLM Configuration
#
ollama_base_url = os.getenv('OLLAMA_BASE_URL')
default_model = os.getenv('OLLAMA_MODEL_DEFAULT')

default_prompt = (
    "You are a helpful assistant. Output answers in Markdown. Use $ and $$ to surround "
    "mathematical formulas."
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
huggingface_model_cache_folder = os.getenv('SENTENCE_TRANSFORMERS_HOME')
huggingface_default_embedding_model = os.getenv('HUGGINGFACE_EMBEDDING_MODEL_DEFAULT')
pdf_parser_model = os.getenv('PDF_PARSER_MODEL')

# Vector store configuration
#
prefer_opensearch = _bool_from_env('PREFER_OPENSEARCH')
opensearch_base_url = os.getenv('OPENSEARCH_BASE_URL')
opensearch_index_prefix = os.getenv('OPENSEARCH_INDEX_PREFIX')


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
