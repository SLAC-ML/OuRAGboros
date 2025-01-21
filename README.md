# Winter 2025 Stanford ICME Research Rotation

## Getting Started

First, [install uv](https://docs.astral.sh/uv/getting-started/installation/) and [Ollama](https://ollama.com/).

Then, start Ollama with

```sh
$ ollama start
```

Then, pull the latest `llama3` model and run this application with `uv`:

```sh
$ ollama pull llama3
$ uv run streamlit run src/main.py
```

...and then navigate to http://localhost:8501 in your favorite browser to get started.
