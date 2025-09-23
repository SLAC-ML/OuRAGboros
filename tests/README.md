# Tests Directory

This directory contains test scripts for various OuRAGboros components.

## Test Files

- `test_qdrant_basic.py` - Basic Qdrant functionality tests
- `test_qdrant_comprehensive.py` - Comprehensive Qdrant integration tests
- `test_qdrant_local.py` - Local Qdrant testing
- `test_qdrant_simple.py` - Simple Qdrant connection tests
- `test_qdrant_ui_integration.py` - UI integration tests with Qdrant
- `test_logging.py` - Auto-log and eval feature testing

## Running Tests

Make sure your services are running first:

```bash
# For Docker Compose testing
./scripts/local-dev.sh up

# Then run tests
python tests/test_qdrant_ui_integration.py
python tests/test_logging.py
```

## Environment Setup

Tests use the same environment variables as the main application. See root `.env` and `.env.local` files for configuration.