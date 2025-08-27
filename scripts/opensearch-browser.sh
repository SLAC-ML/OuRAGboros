#!/bin/bash
# OpenSearch Browser for OuRAGboros
# Usage: ./scripts/opensearch-browser.sh [command]
# Commands: indices, docs, search <term>, sample, cleanup

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Start port forwarding if not already running
start_port_forward() {
    if ! pgrep -f "kubectl port-forward.*opensearch.*9200:9200" > /dev/null; then
        echo -e "${BLUE}Starting OpenSearch port forwarding...${NC}"
        kubectl port-forward -n ouragboros svc/opensearch 9200:9200 > /dev/null 2>&1 &
        sleep 3
    fi
}

# Check if OpenSearch is accessible
check_opensearch() {
    if ! curl -s "localhost:9200" > /dev/null; then
        echo -e "${RED}❌ OpenSearch not accessible. Make sure port forwarding is running.${NC}"
        exit 1
    fi
}

show_indices() {
    echo -e "${GREEN}=== OpenSearch Indices ===${NC}"
    curl -s "localhost:9200/_cat/indices?v&h=index,docs.count,store.size" | grep -E "(ouragboros|top_queries)" || echo "No OuRAGboros indices found"
}

show_docs() {
    echo -e "${GREEN}=== Document Sources ===${NC}"
    curl -s "localhost:9200/ouragboros_768_*/_search?size=100&_source=metadata.source" | python3 -c "
import json, sys
data = json.load(sys.stdin)
sources = {}
for hit in data['hits']['hits']:
    source = hit['_source']['metadata']['source']
    sources[source] = sources.get(source, 0) + 1

total = sum(sources.values())
print(f'Total documents: {total} chunks')
print('Sources:')
for source, count in sorted(sources.items()):
    print(f'  📄 {source}: {count} chunks')
"
}

show_sample() {
    echo -e "${GREEN}=== Sample Documents ===${NC}"
    curl -s "localhost:9200/ouragboros_768_*/_search?size=3&_source=text,metadata.source,metadata.page_number" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for i, hit in enumerate(data['hits']['hits'][:3], 1):
    source = hit['_source']['metadata']['source']
    page = hit['_source']['metadata'].get('page_number', 'N/A')
    text = hit['_source']['text'][:300] + '...' if len(hit['_source']['text']) > 300 else hit['_source']['text']
    print(f'📄 {i}. {source} (page {page})')
    print(f'   {text}')
    print()
"
}

search_docs() {
    local term="$1"
    if [ -z "$term" ]; then
        echo "Usage: $0 search <search_term>"
        exit 1
    fi
    
    echo -e "${GREEN}=== Searching for: '$term' ===${NC}"
    curl -s "localhost:9200/ouragboros_768_*/_search" -H 'Content-Type: application/json' -d "{
        \"size\": 5,
        \"query\": {\"match\": {\"text\": \"$term\"}},
        \"_source\": [\"text\", \"metadata.source\", \"metadata.page_number\"]
    }" | python3 -c "
import json, sys
data = json.load(sys.stdin)
total = data['hits']['total']['value']
print(f'Found {total} matches:')
print()
for i, hit in enumerate(data['hits']['hits'], 1):
    source = hit['_source']['metadata']['source']
    page = hit['_source']['metadata'].get('page_number', 'N/A')
    score = hit['_score']
    text = hit['_source']['text'][:400] + '...' if len(hit['_source']['text']) > 400 else hit['_source']['text']
    print(f'{i}. {source} (page {page}) - Score: {score:.2f}')
    print(f'   {text}')
    print()
"
}

cleanup() {
    echo -e "${YELLOW}Stopping port forwarding...${NC}"
    pkill -f "kubectl port-forward.*opensearch.*9200:9200" || true
}

show_help() {
    echo -e "${BLUE}OpenSearch Browser for OuRAGboros${NC}"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  indices    - Show OpenSearch indices"
    echo "  docs       - Show document sources and counts"
    echo "  sample     - Show sample documents"
    echo "  search <term> - Search documents for a term"
    echo "  cleanup    - Stop port forwarding"
    echo ""
    echo "Examples:"
    echo "  $0 docs"
    echo "  $0 search 'neural network'"
    echo "  $0 sample"
}

# Main
start_port_forward
check_opensearch

case "${1:-help}" in
    "indices")
        show_indices
        ;;
    "docs")
        show_docs
        ;;
    "sample")
        show_sample
        ;;
    "search")
        search_docs "$2"
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|*)
        show_help
        ;;
esac