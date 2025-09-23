# Multi-Model Fine-Tuned Embedding Support

This document describes the multi-model fine-tuned embedding support in OuRAGboros, allowing deployment and use of multiple custom embedding models simultaneously.

## Overview

OuRAGboros now supports automatic discovery and use of multiple fine-tuned embedding models through directory-based scanning, replacing the previous single-model environment variable approach.

## Directory Structure

Fine-tuned models are stored in the `finetuned/` subdirectory of the models cache:

```
models/
├── finetuned/                    # Fine-tuned models directory
│   ├── physics-specialized/      # Custom physics model
│   │   ├── config.json          # Model configuration
│   │   ├── pytorch_model.bin    # Model weights
│   │   ├── tokenizer.json       # Tokenizer files
│   │   └── model_metadata.json  # Optional: Custom display name
│   ├── chemistry-optimized/      # Custom chemistry model
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer.json
│   └── domain-expert/            # Another domain-specific model
│       └── ...
└── models--{org}--{model}/       # HuggingFace cache format
    └── ...
```

## Model Discovery

Models are automatically discovered by scanning the `finetuned/` directory. A valid model must contain:

- `config.json` - HuggingFace model configuration
- Model weights file: `pytorch_model.bin`, `model.safetensors`, or `pytorch_model.safetensors`

## Model Metadata (Optional)

Each model can include a `model_metadata.json` file for custom display names:

```json
{
  "display_name": "Physics Domain Expert",
  "description": "Fine-tuned for physics literature and concepts",
  "training_date": "2025-09-10",
  "version": "1.0"
}
```

If no metadata file exists, display names are generated automatically from directory names.

## Local Development

### Docker Compose Setup

The `docker-compose.local.yml` is configured with a bind mount for local testing:

```yaml
volumes:
  - ./models:/app/models  # Bind mount for local testing
```

### Testing Workflow

1. **Create model directory structure**:
   ```bash
   mkdir -p models/finetuned/my-custom-model
   ```

2. **Copy trained model files**:
   ```bash
   cp /path/to/trained/model/* models/finetuned/my-custom-model/
   ```

3. **Add metadata (optional)**:
   ```bash
   echo '{"display_name": "My Custom Model"}' > models/finetuned/my-custom-model/model_metadata.json
   ```

4. **Restart services**:
   ```bash
   ./scripts/local-dev.sh restart
   ```

5. **Verify in UI**: The model should appear in the embedding model dropdown

## Kubernetes Deployment

### Persistent Volume Configuration

The `ouragboros-models` PVC is configured for 20GB to accommodate multiple models:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ouragboros-models
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
```

### Volume Mount

Models are mounted at `/app/models` in the ouragboros container:

```yaml
volumeMounts:
  - mountPath: /app/models
    name: ouragboros-models
```

### Model Upload to Kubernetes

1. **Copy models to persistent volume**:
   ```bash
   # Create a temporary pod to access the PVC
   kubectl run model-uploader --image=alpine --rm -it --restart=Never \
     --overrides='{
       "spec": {
         "containers": [{
           "name": "model-uploader",
           "image": "alpine",
           "command": ["sleep", "3600"],
           "volumeMounts": [{
             "name": "models",
             "mountPath": "/models"
           }]
         }],
         "volumes": [{
           "name": "models",
           "persistentVolumeClaim": {"claimName": "ouragboros-models"}
         }]
       }
     }' -n ouragboros

   # Copy models to the pod
   kubectl cp ./models/finetuned/my-model ouragboros/model-uploader:/models/finetuned/my-model

   # Cleanup
   kubectl delete pod model-uploader -n ouragboros
   ```

2. **Alternative: Direct kubectl cp to running pod**:
   ```bash
   # Find the ouragboros pod
   POD_NAME=$(kubectl get pods -n ouragboros -l io.kompose.service=ouragboros -o jsonpath='{.items[0].metadata.name}')

   # Copy model directory
   kubectl cp ./models/finetuned/my-model ouragboros/$POD_NAME:/app/models/finetuned/my-model

   # Restart pod to pick up new models
   kubectl rollout restart deployment/ouragboros -n ouragboros
   ```

## Model Training and Export

### Training a Model

Use the existing fine-tuning pipeline in `src/tools/physbert_triplet_finetuning/`:

```bash
# Run the fine-tuning notebook or script
uv run python src/tools/physbert_triplet_finetuning/finetune_transformers.ipynb

# Models are saved to models/physbert-physics-finetuned/ by default
```

### Export for Multi-Model Support

1. **Move to finetuned directory**:
   ```bash
   mkdir -p models/finetuned/physics-expert
   cp -r models/physbert-physics-finetuned/* models/finetuned/physics-expert/
   ```

2. **Add metadata**:
   ```bash
   echo '{
     "display_name": "Physics Domain Expert",
     "description": "Fine-tuned for physics literature",
     "training_date": "'$(date -I)'",
     "base_model": "thellert/physbert_cased"
   }' > models/finetuned/physics-expert/model_metadata.json
   ```

## API Usage

### Listing Available Models

Models appear in the `/ask` endpoint with `huggingface:` prefix:

```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is quantum mechanics?",
    "embedding_model": "huggingface:/app/models/finetuned/physics-expert",
    "llm_model": "stanford:gpt-4",
    "knowledge_base": "physics_papers"
  }'
```

### Model Identifiers

- Base models: `huggingface:thellert/physbert_cased`
- Fine-tuned models: `huggingface:/app/models/finetuned/physics-expert`
- Legacy support: `huggingface:/app/models/ouragboros_finetuned` (if exists)

## Migration from Single Model

### Existing Deployments

The system maintains backward compatibility:

1. **Environment variable support**: `HUGGINGFACE_FINETUNED_EMBEDDING_MODEL` still works
2. **Legacy model detection**: Existing models are automatically detected
3. **Gradual migration**: New models can be added while keeping existing ones

### Migration Steps

1. **Create new directory structure** in existing deployment
2. **Copy existing model** to new location
3. **Update deployment** to remove environment variable (optional)
4. **Add new models** as needed

## Troubleshooting

### Model Not Appearing

1. **Check directory structure**: Ensure `config.json` and weights file exist
2. **Verify permissions**: Models directory must be readable by container
3. **Check logs**: Look for validation errors in application logs
4. **Clear cache**: Restart application to refresh model discovery

### Model Loading Errors

1. **Validate model format**: Ensure model is compatible with HuggingFace transformers
2. **Check disk space**: Ensure sufficient storage for all models
3. **Memory issues**: Consider resource limits for large models

### Performance Considerations

1. **Model size**: Large models consume more memory and disk space
2. **Load time**: Multiple models increase startup time
3. **Caching**: Models are cached after first use to improve performance

## Best Practices

1. **Model naming**: Use descriptive directory names (e.g., `physics-literature-v2`)
2. **Metadata files**: Always include model metadata for better UX
3. **Version control**: Keep track of model versions and training parameters
4. **Storage management**: Monitor disk usage and clean up unused models
5. **Testing**: Validate models in local Docker before K8s deployment