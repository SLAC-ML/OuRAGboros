import os

# Default file search path for documents. This searches the current
# repository for Markdown files.
#
default_root_doc_path = '.'

# LLM Configuration
#
default_model = os.environ.get(
    'OLLAMA_MODEL_DEFAULT',
    default='ollama:llama3.1:latest',
)
default_prompt = """You are an assistant tasked with helping students get acquainted with 
a new research project designed to make sense of a long series of scientific logs 
written by equipment operators at the Stanford Linear Accelerator. If you don't know 
the answer, say you don't know. Keep answers concise. 
Context: {}"""

# Local embeddings cache folder. Only used for HuggingFace embeddings.
#
# See https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.html#langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.cache_folder
huggingface_model_cache_folder = os.getenv(
    'SENTENCE_TRANSFORMERS_HOME',
    default='./models',
)
pdf_parser_model = 'facebook/nougat-small'

# Vectorstore configuration
#
opensearch_url = os.getenv(
    'OPENSEARCH_HOST',
    default='http://127.0.0.1:9200',
)
opensearch_index = os.getenv(
    'OPENSEARCH_INDEX',
    default='ouragborous',
)
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
