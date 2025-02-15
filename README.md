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
```

If you have OpenSearch or Ollama running somewhere other than locally, you can change the
appropriate configuration by modifying the environment variables listed in the 
`.default.env` file at the root of this directory or by modifying your host system's 
environment; the latter configuration takes precedence.

The `.env` file contains configuration to run the project via 
[Docker Compose](https://docs.docker.com/compose/) and should be modified as necessary 
when deploying the application to production. It follows exactly the same format as 
`.default.env`, and any environment variables not specified in `.env` will default to
the values listed in `.default.env`.

Once everything's configured, head to http://localhost:8501 in your favorite browser to 
get started.

This project uses the `llama3.1:latest` LLM by default, but any other model (e.g. 
`deepseek-r1`, `codellama`, etc.) can be used by simply running 
`ollama pull <model name>`. The newly pulled model will be available to select in a 
drop-down on the webpage above after the page is reloaded. Alternatively, you can set the 
`OLLAMA_MODEL_DEFAULT` environment variable to 
[any supported Ollama model name](https://ollama.com/search) to use that model by default. 
If the model is not found locally, it will be pulled when a user prompt has been provided
in the application UI.

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
