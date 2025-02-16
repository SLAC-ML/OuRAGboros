import os

import dotenv

dotenv.load_dotenv('.default.env')


def _bool_from_env(env: str) -> bool:
    value = os.getenv(env)

    true_vals = ('yes', 'true', 't', 'y', '1')
    false_vals = ('no', 'false', 'f', 'n', '0', 'off')
    if value.lower() in true_vals:
        return True
    elif value.lower() in false_vals:
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

default_prompt = """You are a helpful assistant. When possible, use available 
context to answer questions. Keep answers concise. Output answers in Markdown.
"""

# Local embeddings cache folder. Only used for HuggingFace embeddings.
#
# See https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.html#langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.cache_folder
huggingface_model_cache_folder = os.getenv('SENTENCE_TRANSFORMERS_HOME')
pdf_parser_model = os.getenv('PDF_PARSER_MODEL')

# Vector store configuration
#
prefer_opensearch = _bool_from_env('PREFER_OPENSEARCH')
opensearch_base_url = os.getenv('OPENSEARCH_BASE_URL')
opensearch_index_prefix = os.getenv('OPENSEARCH_INDEX_PREFIX')

opensearch_index_settings = lambda vector_size=768: {
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
