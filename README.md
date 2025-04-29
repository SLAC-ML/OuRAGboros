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

### Docker Compose

[Docker Compose](https://docs.docker.com/compose/) is by far the easiest way to run this
project; the `.env` file contains the necessary configuration for this out-of-the-box
and may be modified as necessary when deploying the application to an external server.
The `.env` file follows exactly the same format as `.default.env`, and any environment
variables not specified in `.env` will default to the values listed in `.default.env`.

When run via Docker Compose, this application starts and uses OpenSearch as a persistent 
vector database. Once you've [installed Docker Compose](https://docs.docker.com/compose/install/),
just run

```sh
$ docker-compose up
```

in this directory to build and start the application. 

### Usage

Whether you started the application via `uv` or `docker-compose`, you can head to 
http://localhost:8501 in your favorite browser once everything's running to get started.


#### Kubernetes/Helm

> NOTE: This section contains deployment guidelines for managing this application as a
> [Kubernetes cluster](https://kubernetes.io/docs/home/). If you don't know what that is,
> you're probably safe to skip this section unless you've been assigned the task of 
> deploying this application to a remote server for the multi-user use case.

We use [Kompose](https://kompose.io/) in conjunction with 
[Kustomize](https://kubernetes.io/docs/tasks/manage-kubernetes-objects/kustomization/) to 
convert the `docker-compose.yml` file included in this project to the corresponding 
[Helm chart](https://helm.sh/) (and from there, relevant Kubernetes deployment config).
To generate these files, just run:

```sh
$ kompose convert -c
```

Default values for this deployment can be found in `docker-compose/values.yaml`, which
follows the standard [Helm Values Files format](https://helm.sh/docs/chart_template_guide/values_files/).
This file is unaffected by the `kompose convert` command, so it's safe to modify 
`docker-compose.yml` and re-generate the Helm chart as needed; the `docker-compose.yml`
file is configured to read environment variables from `.kubernetes.env` during chart
generation. To install the chart (which deploys services to a Kubernetes cluster), run:

```sh
$ helm template --values ./docker-compose/values.yaml ./docker-compose > k8s.yaml
$ kubectl create namespace ouragboros --dry-run=client -o yaml | kubectl apply -f -
$ kubectl apply --namespace ouragboros -k .
```

Running these commands will also allow you to upgrade an existing installation.

##### Releasing

When releasing a new application version, you'll need to update the `ouragboros` image
tag so that Kubernetes pulls the new version. You'll also want to 
[tag the release](https://git-scm.com/book/en/v2/Git-Basics-Tagging) to a specific
[semantic version](https://semver.org/). Here are the steps for this:

1) Update `docker-compose.yml` with the new desired release number (e.g., `0.0.2`):

```yaml
  ouragboros:
    build: .
    image: schrodingersket/ouragboros:0.0.2
```

2) Build and push the new Docker images:

```shell
$ docker-compose build --push
```

3) Re-run the `kompose` command from above.
   
4) Commit the updated Helm config changes and tag the new release:

```sh
$ git add docker-compose docker-compose.yml
$ git commit -m '0.0.2'
$ git tag v0.0.2
$ git push --tags
```

5) Deploy the new application version by re-running the `helm upgrade` command from above.

#### Minikube

This Kubernetes deployment was developed and tested using 
[Minikube](https://minikube.sigs.k8s.io/docs/start/). If you're using Minikube, you can
access the main application after installing the Helm chart via:

```sh
$ minikube service ouragboros-tcp --namespace ouragboros
```

#### GPU Acceleration [S3DF Kubernetes Cluster]

For any service that needs access to GPU resources, you'll need to add the following


#### GPU Acceleration [Local Kubernetes Cluster]

See the [official Kubernetes guide](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
for the latest documentation. If using NVIDIA drivers, you can install the [relevant Helm
chart](https://github.com/NVIDIA/k8s-device-plugin?tab=readme-ov-file#deployment-via-helm):

```sh
$ helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
$ helm repo update
$ helm install ouragboros-nvidia-device-plugin nvdp/nvidia-device-plugin --namespace ouragboros --create-namespace
```

If using Minikube, you'll also need to be sure to allow Minikube to access your GPUs:

```sh
$ minikube start --docker-opt="default-ulimit=nofile=102400:102400" --driver docker --container-runtime docker --gpus all
```

### Local Tooling (`uv`, `ollama`) Installation

If you prefer not to use a system-wide installation (or do not have root access), you can
install `uv` and `ollama` locally.

Check the [latest installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
for `uv`; on UNIX systems, you can install a local copy without modifying your default
shell:

```sh
$ curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="./uv/bin" sh
```

Ollama can be installed by 
[following the Ollama Installation Guide](https://ollama.com/download). On Linux, this is as simple as:

```sh
$ mkdir ollama && curl -L https://ollama.com/download/ollama-linux-amd64.tgz | tar -xz -C ollama

# Or, if using an ARM CPU:
#
$ mkdir ollama && curl -L https://ollama.com/download/ollama-linux-arm64.tgz | tar -xz -C ollama

# See https://github.com/ollama/ollama/blob/main/docs/linux.md for more information.
```

If you install `uv` (resp. `ollama`) locally, you will need to replace the above `uv`
(resp. `ollama`) command with `./uv/bin/uv` (resp. `./path/to/ollama`).

### OpenSearch

This application runs with an in-memory vector store by default; this means that any
uploaded embeddings will be lost when the application restarts. To support persistent
embeddings, we use the [OpenSearch](https://opensearch.org/) vector store. To point to
an existing OpenSearch installation, you can change the appropriate configuration by 
modifying the environment variables listed in the `.default.env` file at the root of this 
directory (or by modifying your host system's environment); the latter configuration 
takes precedence.

Check the [OpenSearch downloads](https://opensearch.org/downloads.html) page for detailed
installation information. If you're running on MacOS, you'll need to use Docker Compose
to run OpenSearch (which is recommended for all users anyway).

### Changing LLMs

This project uses the `llama3.1:latest` LLM by default, but any other Ollama model (e.g. 
`deepseek-r1`, `codellama`, etc.) can be used simply by running `ollama pull <model name>` 
(or `docker-compose exec ollama ollama pull <model name>` if using Docker Compose). The 
newly pulled model will be available to select in a drop-down on the webpage above after 
the page is reloaded. Alternatively, you can set the `OLLAMA_MODEL_DEFAULT` environment 
variable to [any supported Ollama model name](https://ollama.com/search) to use that model 
by default. If the model is not found locally, it will be pulled when a user prompt has 
been provided in the application UI.

### Offline PDF Chunking

In addition to in-application PDF chunking, this module provides a mechanism 
to chunk documents on the command-line in order to run offline and take advantage of 
hardware acceleration. To see how to do so, just run `uv run chunk_pdf --help` to see all
command line options:

```
$ uv run chunk_pdf --help
usage: uv run chunk_pdf [-h] [--outpath OUTPATH] filename

Chunks a PDF into a bunch of text files.

positional arguments:
  filename           Input PDF file.

options:
  -h, --help         show this help message and exit
  --outpath OUTPATH  Output directory to store processed text files.
```

For example:

```
$ mkdir -p data/chunks
$ uv run chunk_pdf <pdf path> ./data/chunks
```

### Offline Fine-Tuning

Fine-tuning a sentence transformer can make a big difference in the quality of text 
embeddings. To see how this is done, just run `uv run finetune_model --help` to see all
command line options:

```
$ uv run finetune_model --help
usage: uv run finetune [-h] [--outpath OUTPATH] [--base-model BASE_MODEL] [--tuned-model-name TUNED_MODEL_NAME] [--log-dir LOG_DIR] filename

Fine-tunes a particular HuggingFace model on a body of text.

positional arguments:
  filename              Text file to use for model tuning.

options:
  -h, --help            show this help message and exit
  --outpath OUTPATH     Destination path for the fine-tuned model.
  --base-model BASE_MODEL
                        Base model to fine-tune from.
  --tuned-model-name TUNED_MODEL_NAME
                        Name for the fine-tuned model.
  --log-dir LOG_DIR     Directory in which to store training log files.
```

For example, to fine-tune a model for use in this project, you can run:

```
$ uv run finetune_model <text file path> \
    --outpath=./models \
    --base-model=google-bert/bert-base-uncased \
    --tuned-model-name=my-cool-model
```

Once the model is trained, just update the `HUGGINGFACE_FINETUNED_EMBEDDING_MODEL` 
environment variable in `.default.env` to `my-cool-model`, refresh your browser page, 
and start experimenting!
