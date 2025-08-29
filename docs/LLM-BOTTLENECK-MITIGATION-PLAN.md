# LLM Bottleneck Mitigation Plan

**Created**: August 29, 2025  
**Context**: Following comprehensive profiling that identified LLM token generation as the primary bottleneck (not embedding generation)

## Current Performance Baseline
- **Stanford API (gpt-4o)**: 10.8s single request, degrades to 33s at 3x concurrency
- **OpenAI (gpt-4o-mini)**: 7.6s single request (30% faster)
- **Local Ollama**: 107.5s (not viable for production)
- **RAG Pipeline**: <1s (already optimized with Qdrant)

## Goal
Achieve **3s response time at 100 concurrent users**

## Implementation Phases

### Phase 1: Critical Bug Fixes (Week 1)
**Objective**: Eliminate fallback mode overhead and optimize API integration

1. **Fix Primary LLM Query Bug**
   - Resolve `'str' object has no attribute 'content'` error
   - Eliminate forced fallback mode
   - Expected improvement: 2-3 seconds per request

2. **Optimize ChatOpenAI Integration**
   - Fix streaming token handling for all providers
   - Standardize model name parsing
   - Consistent error handling

3. **Switch Default to OpenAI**
   - Change default from Stanford API to OpenAI gpt-4o-mini
   - 30% immediate performance improvement
   - Better concurrency scaling expected

4. **Implement Connection Pooling**
   - HTTP connection reuse for API calls
   - Reduce connection establishment overhead
   - Configure optimal timeout and retry settings

### Phase 2: Concurrency Architecture (Week 2)
**Objective**: Handle 100 concurrent users efficiently

5. **Async Request Processing**
   - Implement async/await patterns for LLM calls
   - Non-blocking I/O operations
   - Concurrent request processing without blocking

6. **Request Queue Management**
   - Add request queuing for peak loads
   - Implement backpressure handling
   - Graceful degradation under extreme load

7. **Load Testing**
   - Test at 20, 50, 100 concurrent users
   - Identify concurrency bottlenecks
   - Optimize worker configuration

### Phase 3: Performance Enhancements (Week 3)
**Objective**: Achieve sub-3s response times

8. **Response Caching**
   - LLM response caching for identical queries
   - Multi-level cache (embedding + query hash)
   - Sub-second responses for cached queries

9. **Streaming Response Optimization**
   - Stream tokens as generated (see STREAMING-API-IMPLEMENTATION.md)
   - Reduce perceived latency
   - Better user experience

10. **Performance Monitoring**
    - Real-time bottleneck detection
    - Metrics dashboard
    - Alert system for degraded performance

## Expected Performance Improvements

| Phase | Current | Target | Improvement | Key Metric |
|-------|---------|--------|-------------|------------|
| Baseline | 10.8s | - | - | Single user |
| Phase 1 | 10.8s | 5-6s | 50% | Single user |
| Phase 2 | 5-6s | 5-6s | Scaling | 100 concurrent |
| Phase 3 | 5-6s | 1-3s | 80% | Cached + streaming |

## Technical Implementation Details

### Priority 1: Fix Fallback Mode
```python
# Current issue in lib/rag_service.py
# Primary query fails with: 'str' object has no attribute 'content'
# Forces all requests into fallback ChatOpenAI mode
# Need to fix langchain_llm.query_llm() token handling
```

### Priority 2: OpenAI Configuration
```python
# Update config.py defaults
DEFAULT_LLM_MODEL = "openai:gpt-4o-mini"  # From stanford:gpt-4o

# Ensure proper API key configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

### Priority 3: Async Implementation
```python
# Convert synchronous calls to async
async def answer_query_async(...):
    # Use aiohttp for API calls
    # Implement proper async token streaming
    # Handle concurrent requests efficiently
```

### Priority 4: Caching Strategy
```python
# Implement with Redis or in-memory cache
cache_key = hash(query + embedding_model + llm_model + knowledge_base)
if cached_response := cache.get(cache_key):
    return cached_response
```

## Success Metrics

1. **Response Time**: <3s at P50, <5s at P95
2. **Concurrency**: Handle 100 concurrent users
3. **Throughput**: >20 requests/second
4. **Availability**: 99.9% uptime
5. **User Experience**: First token <500ms (with streaming)

## Risk Mitigation

1. **API Rate Limits**: Implement request queuing and backoff
2. **Cost Management**: Monitor API usage and implement quotas
3. **Fallback Strategy**: Multiple LLM providers for redundancy
4. **Cache Invalidation**: TTL-based cache expiry
5. **Security**: Rate limiting per user/IP

## Next Steps

1. ✅ Save this mitigation plan
2. → **Implement Streaming API** (see STREAMING-API-IMPLEMENTATION.md)
3. Execute Phase 1 bug fixes
4. Test and iterate

## Related Documents
- `benchmark/reports/llm-bottleneck-profiling-analysis-2025-08-29.md` - Profiling results
- `docs/STREAMING-API-IMPLEMENTATION.md` - Streaming API design (to be created)