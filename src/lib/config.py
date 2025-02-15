import os

import dotenv

dotenv.load_dotenv('.default.env')

# LLM Configuration
#
ollama_base_url = os.getenv('OLLAMA_BASE_URL')
default_model = os.getenv('OLLAMA_MODEL_DEFAULT')

default_prompt = """You are an assistant tasked with helping students and engineers get 
understand high-level engineering concepts. You use available context to answer 
questions about the material contained in the context. Keep answers concise, but elaborate 
upon request. Output answers in Markdown.
Context: {}"""

# Local embeddings cache folder. Only used for HuggingFace embeddings.
#
# See https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.html#langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.cache_folder
huggingface_model_cache_folder = os.getenv('SENTENCE_TRANSFORMERS_HOME')
pdf_parser_model = os.getenv('PDF_PARSER_MODEL')

# Vectorstore configuration
#
opensearch_base_url = os.getenv('OPENSEARCH_BASE_URL')
opensearch_index_prefix = os.getenv('OPENSEARCH_INDEX_PREFIX')

opensearch_index_settings = lambda vector_size=768:  {
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
