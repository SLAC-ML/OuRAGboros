#!/bin/bash
# Model Management Utility for OuRAGboros Multi-Model Support
#
# This script helps manage fine-tuned embedding models in both local and K8s environments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
FINETUNED_DIR="$MODELS_DIR/finetuned"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
Model Management Utility for OuRAGboros

Usage: $0 <command> [arguments]

Commands:
    list                     List all discovered fine-tuned models
    validate <model_name>    Validate a specific model directory
    create <model_name>      Create a new model directory structure
    upload <model_name>      Upload model to Kubernetes (requires kubectl)
    metadata <model_name>    Create metadata file for a model
    test                     Test model discovery functionality

Examples:
    $0 list                           # List all models
    $0 validate physics-expert        # Validate physics-expert model
    $0 create chemistry-v2            # Create new model directory
    $0 upload physics-expert          # Upload to K8s
    $0 metadata physics-expert        # Create metadata file

EOF
}

info() {
    echo -e "${BLUE}ℹ ${1}${NC}"
}

success() {
    echo -e "${GREEN}✅ ${1}${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  ${1}${NC}"
}

error() {
    echo -e "${RED}❌ ${1}${NC}"
}

ensure_finetuned_dir() {
    if [[ ! -d "$FINETUNED_DIR" ]]; then
        info "Creating finetuned models directory: $FINETUNED_DIR"
        mkdir -p "$FINETUNED_DIR"
    fi
}

list_models() {
    info "Scanning for fine-tuned models in: $FINETUNED_DIR"

    if [[ ! -d "$FINETUNED_DIR" ]]; then
        warning "Finetuned directory not found: $FINETUNED_DIR"
        return 1
    fi

    local count=0
    for model_dir in "$FINETUNED_DIR"/*/; do
        if [[ -d "$model_dir" ]]; then
            local model_name=$(basename "$model_dir")
            local config_file="$model_dir/config.json"
            local weights_file=""

            # Check for weights file
            for weight in "pytorch_model.bin" "model.safetensors" "pytorch_model.safetensors"; do
                if [[ -f "$model_dir/$weight" ]]; then
                    weights_file="$weight"
                    break
                fi
            done

            local status="❌ Invalid"
            if [[ -f "$config_file" && -n "$weights_file" ]]; then
                status="✅ Valid"
            fi

            echo "  $((++count)). $model_name ($status)"
            echo "     Path: $model_dir"

            # Show metadata if available
            local metadata_file="$model_dir/model_metadata.json"
            if [[ -f "$metadata_file" ]]; then
                local display_name=$(jq -r '.display_name // empty' "$metadata_file" 2>/dev/null || echo "")
                if [[ -n "$display_name" ]]; then
                    echo "     Display Name: $display_name"
                fi
            fi

            echo "     Config: $([ -f "$config_file" ] && echo "✅" || echo "❌")"
            echo "     Weights: $([ -n "$weights_file" ] && echo "✅ $weights_file" || echo "❌")"
            echo
        fi
    done

    if [[ $count -eq 0 ]]; then
        warning "No fine-tuned models found"
        info "Use '$0 create <model_name>' to create a new model directory"
    else
        success "Found $count model(s)"
    fi
}

validate_model() {
    local model_name="$1"
    if [[ -z "$model_name" ]]; then
        error "Model name required"
        usage
        return 1
    fi

    local model_dir="$FINETUNED_DIR/$model_name"
    info "Validating model: $model_name"
    info "Model directory: $model_dir"

    if [[ ! -d "$model_dir" ]]; then
        error "Model directory not found: $model_dir"
        return 1
    fi

    local valid=true

    # Check config.json
    local config_file="$model_dir/config.json"
    if [[ -f "$config_file" ]]; then
        success "config.json found"
        # Validate JSON
        if jq empty "$config_file" 2>/dev/null; then
            success "config.json is valid JSON"
        else
            error "config.json is invalid JSON"
            valid=false
        fi
    else
        error "config.json missing"
        valid=false
    fi

    # Check for weights file
    local weights_found=false
    for weight in "pytorch_model.bin" "model.safetensors" "pytorch_model.safetensors"; do
        if [[ -f "$model_dir/$weight" ]]; then
            success "Model weights found: $weight"
            weights_found=true
            break
        fi
    done

    if [[ "$weights_found" != true ]]; then
        error "No model weights file found (pytorch_model.bin, *.safetensors)"
        valid=false
    fi

    # Check optional files
    local tokenizer_files=("tokenizer.json" "tokenizer_config.json" "vocab.txt")
    for file in "${tokenizer_files[@]}"; do
        if [[ -f "$model_dir/$file" ]]; then
            success "Optional file found: $file"
        fi
    done

    # Check metadata
    local metadata_file="$model_dir/model_metadata.json"
    if [[ -f "$metadata_file" ]]; then
        if jq empty "$metadata_file" 2>/dev/null; then
            success "model_metadata.json found and valid"
            local display_name=$(jq -r '.display_name // empty' "$metadata_file" 2>/dev/null)
            if [[ -n "$display_name" ]]; then
                info "Display name: $display_name"
            fi
        else
            warning "model_metadata.json found but invalid JSON"
        fi
    else
        info "model_metadata.json not found (optional)"
    fi

    if [[ "$valid" == true ]]; then
        success "Model $model_name is valid"
        return 0
    else
        error "Model $model_name is invalid"
        return 1
    fi
}

create_model() {
    local model_name="$1"
    if [[ -z "$model_name" ]]; then
        error "Model name required"
        usage
        return 1
    fi

    # Validate model name (alphanumeric, hyphens, underscores only)
    if [[ ! "$model_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        error "Invalid model name. Use only letters, numbers, hyphens, and underscores."
        return 1
    fi

    ensure_finetuned_dir

    local model_dir="$FINETUNED_DIR/$model_name"
    if [[ -d "$model_dir" ]]; then
        error "Model directory already exists: $model_dir"
        return 1
    fi

    info "Creating model directory: $model_dir"
    mkdir -p "$model_dir"

    # Create basic config.json template
    cat > "$model_dir/config.json" << EOF
{
  "model_type": "bert",
  "hidden_size": 768,
  "vocab_size": 28996,
  "max_position_embeddings": 512,
  "architectures": ["BertModel"]
}
EOF

    success "Created config.json template"

    # Create empty weights file as placeholder
    touch "$model_dir/pytorch_model.bin"
    success "Created placeholder weights file"

    # Create metadata template
    create_metadata "$model_name" true

    success "Model directory created: $model_dir"
    warning "Remember to replace the placeholder files with your actual trained model!"
    info "Next steps:"
    info "  1. Copy your trained model files to: $model_dir"
    info "  2. Update the metadata: $0 metadata $model_name"
    info "  3. Validate the model: $0 validate $model_name"
}

create_metadata() {
    local model_name="$1"
    local is_template="${2:-false}"

    if [[ -z "$model_name" ]]; then
        error "Model name required"
        return 1
    fi

    local model_dir="$FINETUNED_DIR/$model_name"
    if [[ ! -d "$model_dir" ]]; then
        error "Model directory not found: $model_dir"
        info "Use '$0 create $model_name' to create the model directory first"
        return 1
    fi

    local metadata_file="$model_dir/model_metadata.json"
    local display_name

    if [[ "$is_template" == true ]]; then
        # Auto-generate display name for template
        display_name=$(echo "$model_name" | sed 's/[-_]/ /g' | sed 's/\b\w/\U&/g')
        display_name="$display_name (Fine-tuned)"
    else
        # Interactive mode
        echo "Creating metadata for model: $model_name"
        read -p "Display name [$model_name]: " display_name
        display_name="${display_name:-$model_name}"
    fi

    cat > "$metadata_file" << EOF
{
  "display_name": "$display_name",
  "description": "Custom fine-tuned embedding model",
  "created_date": "$(date -I)",
  "version": "1.0",
  "base_model": "thellert/physbert_cased"
}
EOF

    success "Created metadata file: $metadata_file"
    info "Edit the file to customize description and other fields"
}

upload_to_k8s() {
    local model_name="$1"
    if [[ -z "$model_name" ]]; then
        error "Model name required"
        usage
        return 1
    fi

    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
        return 1
    fi

    local model_dir="$FINETUNED_DIR/$model_name"
    if [[ ! -d "$model_dir" ]]; then
        error "Model directory not found: $model_dir"
        return 1
    fi

    # Validate model first
    if ! validate_model "$model_name"; then
        error "Model validation failed. Cannot upload invalid model."
        return 1
    fi

    info "Uploading model $model_name to Kubernetes..."

    # Get ouragboros pod
    local pod_name
    pod_name=$(kubectl get pods -n ouragboros -l io.kompose.service=ouragboros -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "$pod_name" ]]; then
        error "No ouragboros pod found in namespace 'ouragboros'"
        info "Make sure the ouragboros deployment is running"
        return 1
    fi

    info "Found pod: $pod_name"
    info "Copying model files..."

    # Copy model directory to pod
    kubectl cp "$model_dir" "ouragboros/$pod_name:/app/models/finetuned/$model_name"

    if [[ $? -eq 0 ]]; then
        success "Model uploaded successfully"
        info "Restarting deployment to pick up new model..."
        kubectl rollout restart deployment/ouragboros -n ouragboros
        success "Deployment restarted. Model should be available shortly."
    else
        error "Failed to upload model"
        return 1
    fi
}

test_discovery() {
    info "Testing model discovery functionality..."

    # Check if Python test file exists
    local test_file="$REPO_ROOT/test_model_discovery.py"
    if [[ ! -f "$test_file" ]]; then
        error "Test file not found: $test_file"
        return 1
    fi

    cd "$REPO_ROOT"

    # Try to run with uv first, fallback to python
    if command -v uv &> /dev/null; then
        info "Running test with uv..."
        if uv run python test_model_discovery.py; then
            success "Model discovery test passed"
        else
            warning "Test failed with uv, this may be due to missing dependencies"
            info "Try running in a Docker container instead"
        fi
    else
        warning "uv not found, skipping Python test"
        info "Install uv or run tests in Docker container"
    fi
}

main() {
    if [[ $# -eq 0 ]]; then
        usage
        return 1
    fi

    case "$1" in
        list)
            list_models
            ;;
        validate)
            validate_model "$2"
            ;;
        create)
            create_model "$2"
            ;;
        upload)
            upload_to_k8s "$2"
            ;;
        metadata)
            create_metadata "$2"
            ;;
        test)
            test_discovery
            ;;
        *)
            error "Unknown command: $1"
            usage
            return 1
            ;;
    esac
}

main "$@"