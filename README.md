# Winter 2025 Stanford ICME Research Rotation

## Getting Started

First, [install uv](https://docs.astral.sh/uv/getting-started/installation/) and [Ollama](https://ollama.com/).

Then, start Ollama with

```sh
$ ollama start
```

Then, run this application with `uv`:

```sh
$ uv run streamlit run src/main.py
```

...and navigate to http://localhost:8501 in your favorite browser to get started.

This project uses `deepseek-r1` by default, but any other model can be used by simply 
running `ollama pull <model name>`. The newly pulled model will be available to select
in a drop-down on the webpage above. Alternatively, you can set the `OLLAMA_MODEL` 
environment variable to [any supported Ollama model name](https://ollama.com/search) to
use that model by default. If the model is not found locally, it will be pulled when a 
user prompt has been given.
