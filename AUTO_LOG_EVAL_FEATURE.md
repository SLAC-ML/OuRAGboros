# Auto-Log and Eval Feature Documentation

## Overview

The Auto-Log and Eval feature automatically captures, logs, and evaluates all RAG API interactions in OuRAGboros. This enables comprehensive analysis of query performance, model effectiveness, and system behavior without impacting user experience.

## Architecture

### Key Components

1. **QueryLoggerService** (`src/lib/query_logger.py`)
   - Async service with batched OpenSearch logging
   - Non-blocking operation - user responses are never delayed
   - Automatic RAGAS evaluation in background

2. **API Integration** (`src/app_api.py`)
   - Modified `/ask/stream` endpoint captures all interaction data
   - Fire-and-forget logging after streaming response completes
   - Error handling ensures logging failures don't break user experience

3. **RAGAS Evaluator Service** (`docker-compose.yml`)
   - Standalone Docker service for query evaluation
   - Uses Stanford AI API for evaluation (faithfulness, answer_relevancy)
   - Healthcheck monitoring and automatic restarts

4. **Query Logs Viewer** (`query_logs_viewer.html`)
   - Fast, minimal HTML tool for log inspection
   - Real-time search and filtering capabilities
   - Export functionality (JSON/CSV)

### Data Flow

```
User Query → /ask/stream → [Streaming Response to User]
                     ↓
              [Background Logging]
                     ↓
         OpenSearch ← QueryLoggerService → RAGAS Evaluator
                     ↓
               Query Logs Viewer
```

## Quick Start

### 1. Build and Start Services

```bash
# Build the RAGAS evaluator image (from ~/Projects/LLMTest/docker/)
cd ~/Projects/LLMTest/docker
./build.sh

# Start all services including RAGAS evaluator
cd ~/Projects/OuRAGboros
docker-compose up -d
```

### 2. Verify Health

```bash
# Check RAGAS evaluator
curl http://localhost:8002/health

# Check OuRAGboros logging health
curl http://localhost:8001/logs/health
```

### 3. Test Logging

```bash
# Make a query through the streaming API
curl -X POST http://localhost:8001/ask/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is quantum mechanics?",
    "embedding_model": "huggingface:sentence-transformers/all-mpnet-base-v2",
    "llm_model": "stanford:gpt-4",
    "knowledge_base": "default",
    "use_rag": true,
    "use_opensearch": true,
    "prompt": "You are a helpful physics assistant.",
    "max_documents": 5
  }'
```

### 4. View Logs

Open `query_logs_viewer.html` in your browser and set the API endpoint to `http://localhost:8001`.

## Configuration

### Environment Variables

Add to your `.local.env` file:

```bash
# RAGAS Evaluator Configuration
RAGAS_BASE_URL=http://ragas-evaluator:8000

# Stanford AI API (used by RAGAS evaluator)
STANFORD_API_KEY=your_stanford_api_key
STANFORD_BASE_URL=https://aiapi-prod.stanford.edu/v1
```

### OpenSearch Index

Logs are stored in monthly rotating indices:
- Format: `ouragboros_query_logs_YYYY_MM`
- Example: `ouragboros_query_logs_2025_01`
- Automatically created with optimized mapping for search

## API Endpoints

### Query Logging

- **POST** `/logs/search` - Search logs with filters
- **GET** `/logs/stats` - Get aggregated statistics
- **GET** `/logs/export` - Export logs (JSON/CSV)
- **GET** `/logs/health` - Health check for logging service

### Example API Calls

```bash
# Search recent logs
curl -X POST http://localhost:8001/logs/search \
  -H "Content-Type: application/json" \
  -d '{"query": "quantum", "size": 10}'

# Get statistics
curl http://localhost:8001/logs/stats

# Export as CSV
curl "http://localhost:8001/logs/export?format=csv&size=100"
```

## Log Entry Structure

Each logged query contains:

```json
{
  "id": "uuid4",
  "timestamp": "2025-01-15T10:30:00Z",
  "user_query": "What is quantum mechanics?",
  "metadata": {
    "embedding_model": "huggingface:sentence-transformers/all-mpnet-base-v2",
    "llm_model": "stanford:gpt-4",
    "knowledge_base": "physics_papers",
    "score_threshold": 0.5,
    "max_documents": 5,
    "use_opensearch": true,
    "use_qdrant": false,
    "prompt_template": "You are a helpful physics assistant..."
  },
  "rag_results": {
    "documents_count": 3,
    "documents": [
      {"id": "doc1", "score": 0.85, "snippet": "...", "source": "file.pdf"}
    ],
    "retrieval_time": 0.342
  },
  "llm_response": {
    "final_answer": "Quantum mechanics is...",
    "response_time": 2.15,
    "token_count": 245
  },
  "ragas_evaluation": {
    "faithfulness": 0.92,
    "answer_relevancy": 0.87,
    "evaluation_time": 1.23,
    "evaluated_at": "2025-01-15T10:30:03Z"
  },
  "performance_metrics": {
    "total_time": 3.89,
    "embedding_time": 0.12
  },
  "status": "completed"
}
```

## Query Logs Viewer Features

### Search & Filtering
- **Text Search**: Search in questions and answers
- **Date Range**: Filter by timestamp
- **Model Filter**: Filter by LLM model
- **Knowledge Base**: Filter by specific KB
- **Status Filter**: Filter by completion status

### Statistics Dashboard
- Total queries processed
- Average response times
- Average RAGAS scores
- Model usage distribution
- Knowledge base distribution

### Export Options
- **JSON Export**: Full structured data
- **CSV Export**: Flattened data for analysis
- **Filtered Export**: Export only filtered results

### Real-time Features
- Auto-refresh every 30 seconds
- Health status monitoring
- Responsive table with expandable details

## Performance Characteristics

### Non-Blocking Design
- **User Impact**: < 1ms overhead on streaming responses
- **Batch Processing**: Logs flushed every 5 seconds or 10 entries
- **Queue Management**: Max 1000 pending logs with overflow protection

### Concurrent Handling
- **Target Load**: Optimized for ~10 concurrent users
- **OpenSearch Batching**: Reduces individual request overhead
- **Connection Pooling**: Max 20 connections to OpenSearch
- **Circuit Breaker**: Fails gracefully on OpenSearch issues

### Evaluation Performance
- **RAGAS Metrics**: faithfulness, answer_relevancy
- **Evaluation Time**: Typically 1-3 seconds per query
- **Parallel Processing**: Multiple evaluations run concurrently
- **Error Handling**: Evaluation failures don't block logging

## Troubleshooting

### Common Issues

#### 1. RAGAS Evaluation Failing
```bash
# Check RAGAS service health
curl http://localhost:8000/health

# Check RAGAS logs
docker logs ouragboros-ragas-evaluator-1

# Common fix: restart RAGAS service
docker-compose restart ragas-evaluator
```

#### 2. Logs Not Appearing
```bash
# Check logging service health
curl http://localhost:8001/logs/health

# Check OuRAGboros logs
docker logs ouragboros-ouragboros-1

# Verify OpenSearch is running
curl http://localhost:9200/_cluster/health
```

#### 3. Query Logs Viewer Not Loading
- Ensure API endpoint is correct: `http://localhost:8001`
- Check browser console for CORS errors
- Verify services are running with `docker-compose ps`

### Log Levels

Adjust logging verbosity in `src/lib/query_logger.py`:

```python
# Set to DEBUG for detailed logging
logging.getLogger("query_logger").setLevel(logging.DEBUG)
```

### Memory Usage

Monitor queue sizes and OpenSearch memory:

```bash
# Check OpenSearch status
curl http://localhost:9200/_cat/indices/ouragboros_query_logs_*

# Check container memory usage
docker stats ouragboros-opensearch-1
```

## Testing

### Unit Tests

```bash
# Run basic functionality tests
python test_logging.py
```

### Integration Tests

```bash
# Test with actual services (requires docker-compose up)
curl -X POST http://localhost:8001/ask/stream -d '{...}' # Make a query
sleep 10  # Wait for background processing
curl http://localhost:8001/logs/search -d '{"query":"*","size":1}' # Check logs
```

### Load Testing

```bash
# Test concurrent logging (requires services running)
for i in {1..20}; do
  curl -X POST http://localhost:8001/ask/stream -d '{...}' &
done
wait
```

## Monitoring

### Key Metrics to Monitor

1. **Response Times**
   - User streaming response time (should be unaffected)
   - Background logging flush time
   - RAGAS evaluation time

2. **Queue Health**
   - Pending log entries in queue
   - Queue overflow events
   - Batch processing frequency

3. **Storage**
   - OpenSearch index size growth
   - Monthly log volume
   - Disk space usage

4. **Evaluation Success Rate**
   - RAGAS evaluation completion rate
   - Common evaluation errors
   - Stanford AI API quota usage

### Health Checks

The system includes comprehensive health monitoring:

- **RAGAS Service**: `/health` endpoint with curl-based healthcheck
- **Logging Service**: `/logs/health` with OpenSearch connectivity test
- **Background Processing**: Queue size and batch processing monitoring

## Future Enhancements

### Potential Improvements

1. **Additional RAGAS Metrics**
   - Add support for `context_precision` and `context_recall` with reference data
   - Custom evaluation metrics specific to physics domain

2. **Advanced Analytics**
   - Query similarity analysis
   - Model performance comparison dashboard
   - Automated performance alerting

3. **Data Retention**
   - Automated log archival after N months
   - Compressed storage for historical data
   - Data lifecycle management

4. **Real-time Dashboard**
   - Live query monitoring
   - Performance metrics visualization
   - System health dashboard

### Configuration Options

Future configurable parameters:

```python
QUERY_LOGGER_CONFIG = {
    "batch_size": 10,           # Entries per batch
    "batch_interval": 5.0,      # Seconds between flushes
    "max_queue_size": 1000,     # Max pending entries
    "evaluation_timeout": 60,   # RAGAS timeout
    "retry_attempts": 3,        # Failed evaluation retries
    "index_rotation": "monthly" # Index rotation frequency
}
```

## Security Considerations

### Data Privacy
- **Query Content**: User queries and responses are logged - consider data sensitivity
- **API Keys**: RAGAS service uses Stanford AI API - secure key management required
- **Access Control**: Query logs viewer has no authentication - implement access controls for production

### Network Security
- **Internal Services**: RAGAS evaluator only accessible within Docker network
- **CORS Policy**: API endpoints allow all origins - restrict in production
- **HTTPS**: Use HTTPS for production deployments

### Data Retention
- **Log Lifecycle**: Implement data retention policies
- **PII Scrubbing**: Consider scrubbing sensitive information from logs
- **Compliance**: Ensure logging meets organizational data policies

## Support

For issues or questions:

1. Check this documentation
2. Review `test_logging.py` output for diagnostic information
3. Check Docker container logs: `docker-compose logs -f`
4. Verify service health endpoints
5. Review OpenSearch indices: `curl http://localhost:9200/_cat/indices`

The auto-log and eval feature is designed to be robust and non-intrusive, providing valuable insights into RAG system performance while maintaining excellent user experience.