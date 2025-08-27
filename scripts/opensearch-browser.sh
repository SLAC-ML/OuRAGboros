#!/bin/bash
# OpenSearch Browser for OuRAGboros
# Usage: ./scripts/opensearch-browser.sh [command] [knowledge_base]
# Commands: indices, kbs, docs [kb], count [kb], search <term> [kb], sample [kb], cleanup

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

show_knowledge_bases() {
    echo -e "${GREEN}=== Available Knowledge Bases ===${NC}"
    curl -s "localhost:9200/_cat/indices?h=index,creation.date&format=json" | python3 -c "
import sys, json
indices_info = json.load(sys.stdin)
kb_timestamps = {}

for idx_info in indices_info:
    index = idx_info['index']
    if index.startswith('ouragboros_'):
        creation_time = float(idx_info['creation.date'])
        remaining = index[len('ouragboros_'):]
        parts = remaining.split('_')
        
        # If starts with a number, it's the default knowledge base
        if parts[0].isdigit():
            kb_name = 'default'
        else:
            # Find where the vector size starts (first numeric part after kb name)
            kb_parts = []
            for part in parts:
                if part.isdigit():
                    break
                kb_parts.append(part)
            
            if kb_parts:
                kb_name = '_'.join(kb_parts)
            else:
                kb_name = None
        
        # Use the earliest timestamp for each knowledge base (in case of multiple indices)
        if kb_name and (kb_name not in kb_timestamps or creation_time < kb_timestamps[kb_name]):
            kb_timestamps[kb_name] = creation_time

if kb_timestamps:
    print('Available knowledge bases:')
    # Sort by creation time, but ensure default is first
    sorted_kbs = sorted(kb_timestamps.items(), key=lambda x: x[1])
    kb_list = [kb for kb, _ in sorted_kbs]
    
    if 'default' in kb_list:
        kb_list.remove('default')
        kb_list.insert(0, 'default')
    
    for kb in kb_list:
        print(f'  📚 {kb}')
else:
    print('No knowledge bases found')
"
}

# Get index pattern for a specific knowledge base
get_default_index() {
    # Find the default index (format: ouragboros_<number>_<hash>)
    curl -s "localhost:9200/_cat/indices?h=index" | grep "^ouragboros_" | python3 -c "
import sys
for line in sys.stdin:
    index = line.strip()
    parts = index.split('_')
    if len(parts) >= 3 and parts[0] == 'ouragboros' and parts[1].isdigit():
        print(index)
        break
"
}

get_index_pattern() {
    local kb="$1"
    if [ "$kb" = "default" ]; then
        # Get the specific default index name
        get_default_index
    elif [ -z "$kb" ]; then
        echo "ouragboros_*_*"
    else
        echo "ouragboros_${kb}_*"
    fi
}

show_docs() {
    local kb="$1"
    local pattern=$(get_index_pattern "$kb")
    
    if [ -n "$kb" ]; then
        echo -e "${GREEN}=== Document Sources in Knowledge Base: $kb ===${NC}"
    else
        echo -e "${GREEN}=== Document Sources (All Knowledge Bases) ===${NC}"
    fi
    
    curl -s "localhost:9200/${pattern}/_search?size=100&_source=metadata" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'hits' not in data or not data['hits']['hits']:
        print('No documents found')
        sys.exit()
        
    sources = {}
    for hit in data['hits']['hits']:
        source = hit['_source']['metadata']['source']
        sources[source] = sources.get(source, 0) + 1

    total = sum(sources.values())
    print(f'Total documents: {total} chunks')
    print('Sources:')
    for source, count in sorted(sources.items()):
        print(f'  📄 {source}: {count} chunks')
except Exception as e:
    print(f'Error processing documents: {e}')
"
}

count_entries() {
    local kb="$1"
    local pattern=$(get_index_pattern "$kb")
    
    if [ -n "$kb" ]; then
        echo -e "${GREEN}=== Entry Count for Knowledge Base: $kb ===${NC}"
    else
        echo -e "${GREEN}=== Entry Count (All Knowledge Bases) ===${NC}"
    fi
    
    curl -s "localhost:9200/${pattern}/_count" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    count = data.get('count', 0)
    if count == 0:
        print('📊 No entries found')
    else:
        print(f'📊 Total entries: {count:,}')
except Exception as e:
    print(f'Error getting count: {e}')
"
}

show_sample() {
    local kb="$1"
    local pattern=$(get_index_pattern "$kb")
    
    if [ -n "$kb" ]; then
        echo -e "${GREEN}=== Sample Documents from Knowledge Base: $kb ===${NC}"
    else
        echo -e "${GREEN}=== Sample Documents (All Knowledge Bases) ===${NC}"
    fi
    
    curl -s "localhost:9200/${pattern}/_search?size=3&_source=text,metadata.source,metadata.page_number" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'hits' not in data or not data['hits']['hits']:
        print('No documents found')
        sys.exit()
        
    for i, hit in enumerate(data['hits']['hits'][:3], 1):
        source = hit['_source']['metadata']['source']
        page = hit['_source']['metadata'].get('page_number', 'N/A')
        text = hit['_source']['text'][:300] + '...' if len(hit['_source']['text']) > 300 else hit['_source']['text']
        print(f'📄 {i}. {source} (page {page})')
        print(f'   {text}')
        print()
except Exception as e:
    print(f'Error processing documents: {e}')
"
}

search_docs() {
    local term="$1"
    local kb="$2"
    if [ -z "$term" ]; then
        echo "Usage: $0 search <search_term> [knowledge_base]"
        exit 1
    fi
    
    local pattern=$(get_index_pattern "$kb")
    
    if [ -n "$kb" ]; then
        echo -e "${GREEN}=== Searching for: '$term' in Knowledge Base: $kb ===${NC}"
    else
        echo -e "${GREEN}=== Searching for: '$term' (All Knowledge Bases) ===${NC}"
    fi
    
    curl -s "localhost:9200/${pattern}/_search" -H 'Content-Type: application/json' -d "{
        \"size\": 5,
        \"query\": {\"match\": {\"text\": \"$term\"}},
        \"_source\": [\"text\", \"metadata.source\", \"metadata.page_number\"]
    }" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'hits' not in data:
        print('Error: Invalid response from OpenSearch')
        sys.exit()
        
    total = data['hits']['total']['value'] if isinstance(data['hits']['total'], dict) else data['hits']['total']
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
        
    if not data['hits']['hits']:
        print('No matches found')
except Exception as e:
    print(f'Error processing search results: {e}')
"
}

cleanup() {
    echo -e "${YELLOW}Stopping port forwarding...${NC}"
    pkill -f "kubectl port-forward.*opensearch.*9200:9200" || true
}

show_help() {
    echo -e "${BLUE}OpenSearch Browser for OuRAGboros${NC}"
    echo ""
    echo "Usage: $0 [command] [knowledge_base]"
    echo ""
    echo "Commands:"
    echo "  indices           - Show OpenSearch indices"
    echo "  kbs               - List available knowledge bases"
    echo "  docs [kb]         - Show document sources and counts"
    echo "  count [kb]        - Show total entry count for knowledge base"
    echo "  sample [kb]       - Show sample documents"
    echo "  search <term> [kb] - Search documents for a term"
    echo "  cleanup           - Stop port forwarding"
    echo ""
    echo "Knowledge Base Support:"
    echo "  - Omit [kb] to search across all knowledge bases"
    echo "  - Use 'default' for the default knowledge base"
    echo "  - Use custom names like 'physics_papers', 'legal_docs', etc."
    echo ""
    echo "Examples:"
    echo "  $0 kbs                           # List knowledge bases"
    echo "  $0 docs                          # Show all documents"
    echo "  $0 docs default                  # Show documents in default KB"
    echo "  $0 docs physics_papers           # Show documents in physics_papers KB"
    echo "  $0 count                         # Count entries across all KBs"
    echo "  $0 count default                 # Count entries in default KB"
    echo "  $0 search 'neural network'       # Search all knowledge bases"
    echo "  $0 search 'quantum' physics_papers # Search only physics_papers KB"
    echo "  $0 sample default                # Sample from default KB"
}

# Main
start_port_forward
check_opensearch

case "${1:-help}" in
    "indices")
        show_indices
        ;;
    "kbs"|"kb")
        show_knowledge_bases
        ;;
    "docs")
        show_docs "$2"
        ;;
    "count")
        count_entries "$2"
        ;;
    "sample")
        show_sample "$2"
        ;;
    "search")
        search_docs "$2" "$3"
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|*)
        show_help
        ;;
esac