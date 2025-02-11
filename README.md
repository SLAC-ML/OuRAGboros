# Winter 2025 Stanford ICME Research Rotation

## Getting Started

First, [install uv](https://docs.astral.sh/uv/getting-started/installation/) and 
[Ollama](https://ollama.com/). Once installed, start Ollama:

```sh
$ ollama start
```

Then, run this application with `uv`:

```sh
$ uv run streamlit run src/main.py

# If you have Ollama running somewhere other than http://localhost:11434, you can set
# the OLLAMA_BASE_URL environment variable to use that Ollama instance instead:
#
$ OLLAMA_BASE_URL=http://localhost:11434 uv run streamlit run src/main.py
```

...and navigate to http://localhost:8501 in your favorite browser to get started.

This project uses `llama3.1:latest` by default, but any other model (e.g. `deepseek-r1`, 
`codellama`, etc.) can be used by simply running `ollama pull <model name>`. The newly 
pulled model will be available to select in a drop-down on the webpage above after the
page is reloaded. Alternatively, you can set the `OLLAMA_MODEL_DEFAULT` environment 
variable to [any supported Ollama model name](https://ollama.com/search) to use that model 
by default. If the model is not found locally, it will be pulled when a user prompt has 
been provided.

### Local Installation
If you're on UNIX and would like a one-liner to install a local copy of `uv` without
modifying your shell profile, you can run:

```sh
$ curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="./uv/bin" sh
```

Similarly, Ollama can be installed locally with:

```sh
$ mkdir ollama && curl -L https://ollama.com/download/ollama-linux-amd64.tgz | tar -xz -C ollama

# Or, if using an ARM CPU:
#
$ mkdir ollama && curl -L https://ollama.com/download/ollama-linux-arm64.tgz | tar -xz -C ollama

# See https://github.com/ollama/ollama/blob/main/docs/linux.md for more information.
```

If you install `uv` (resp. `ollama`) locally, you will need to replace the above `uv`
(resp. `ollama`) command with `./uv/bin/uv` (resp. `./ollama/bin/ollama`).

### Offline PDF Chunking

In addition to in-application PDF chunking, this module provides a mechanism 
to chunk documents on the command-line in order to run offline and take advantage of 
hardware acceleration. To do so, you can simply run:

```
$ mkdir -p data/chunks
$ uv run chunk_pdf <pdf path> ./data/chunks
```
