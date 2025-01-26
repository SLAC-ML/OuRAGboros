import os

# LLM Configuration
#
default_model = os.environ.get('OLLAMA_MODEL_DEFAULT', default='deepseek-r1:latest')
default_prompt = """You are an assistant tasked with helping students get acquainted with 
a new research project designed to make sense of a long series of scientific logs 
written by equipment operators at the Stanford Linear Accelerator. If you don't know 
the answer, say you don't know. Keep answers concise. Encourage students to reach out 
to the listed collaborators.
Context: {}"""

# Vectorstore configuration
#
opensearch_url = os.getenv('OPENSEARCH_HOST', default='http://127.0.0.1:9200')

# Default file search path for documents
#
default_root_doc_path = '/work/jonathan/stanford/stanford-academics/StanfordNotes/Research/SLAC LLM (Darve Winter 2025)'


#
huggingface_embeddings_cache_folder = './models'

