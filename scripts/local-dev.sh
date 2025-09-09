#!/bin/bash

# OuRAGboros Local Development Script
# Quick rebuild and run for local development with Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default compose file
COMPOSE_FILE="docker-compose.local.yml"

# Help function
show_help() {
    echo -e "${BLUE}OuRAGboros Local Development Script${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       - Rebuild ouragboros container only (default)"
    echo "  build-all   - Rebuild all containers"
    echo "  up          - Start all services"
    echo "  down        - Stop all services"
    echo "  restart     - Rebuild and restart ouragboros"
    echo "  logs        - Show service logs"
    echo "  clean       - Stop services and clean up containers/images"
    echo "  health      - Check health of all services"
    echo ""
    echo "Options:"
    echo "  --no-cache  - Build without Docker cache"
    echo "  -h, --help  - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Rebuild ouragboros and start services"
    echo "  $0 build --no-cache   # Rebuild ouragboros without cache"
    echo "  $0 restart            # Quick rebuild and restart"
    echo "  $0 logs               # Show logs for debugging"
    echo ""
    echo "Services will be available at:"
    echo "  • Streamlit UI:    http://localhost:8501"
    echo "  • REST API:        http://localhost:8001"
    echo "  • Query Logs:      tools/query_logs_viewer.html (set endpoint to localhost:8001)"
    echo "  • RAGAS Evaluator: http://localhost:8002"
    echo "  • OpenSearch:      http://localhost:9200"
    echo "  • Qdrant:          http://localhost:6333"
    echo "  • Ollama:          http://localhost:11434"
}

# Check prerequisites
check_prerequisites() {
    echo -e "${BLUE}🔍 Checking prerequisites...${NC}"
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running. Please start Docker Desktop first.${NC}"
        exit 1
    fi
    echo "✓ Docker is running"
    
    # Check if compose file exists
    if [ ! -f "$COMPOSE_FILE" ]; then
        echo -e "${RED}❌ $COMPOSE_FILE not found.${NC}"
        exit 1
    fi
    echo "✓ $COMPOSE_FILE found"
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}⚠️ .env file not found. Using defaults from docker-compose.local.yml${NC}"
    else
        echo "✓ .env file found"
    fi
    
    echo ""
}

# Build containers
build_containers() {
    local no_cache_flag=""
    local build_target="ouragboros"
    
    if [[ "$1" == "all" ]]; then
        build_target=""
        echo -e "${BLUE}🏗️ Building all containers...${NC}"
    else
        echo -e "${BLUE}🏗️ Building ouragboros container...${NC}"
    fi
    
    if [[ "$2" == "--no-cache" ]] || [[ "$NO_CACHE" == "true" ]]; then
        no_cache_flag="--no-cache"
        echo -e "${YELLOW}Using --no-cache flag${NC}"
    fi
    
    docker compose -f "$COMPOSE_FILE" build $no_cache_flag $build_target
    echo -e "${GREEN}✅ Build completed successfully${NC}"
    echo ""
}

# Start services
start_services() {
    echo -e "${BLUE}🚀 Starting services...${NC}"
    docker compose -f "$COMPOSE_FILE" up -d
    
    echo -e "${GREEN}✅ Services started successfully${NC}"
    echo ""
    show_service_status
}

# Show service status
show_service_status() {
    echo -e "${BLUE}📊 Service Status:${NC}"
    docker compose -f "$COMPOSE_FILE" ps
    echo ""
    echo -e "${GREEN}🌐 Access URLs:${NC}"
    echo "  • Streamlit UI:    http://localhost:8501"
    echo "  • REST API:        http://localhost:8001"
    echo "  • Query Logs:      tools/query_logs_viewer.html (set endpoint to localhost:8001)"
    echo "  • RAGAS Evaluator: http://localhost:8002"
    echo "  • OpenSearch:      http://localhost:9200"
    echo "  • Qdrant:          http://localhost:6333"
    echo "  • Ollama:          http://localhost:11434"
    echo ""
}

# Stop services
stop_services() {
    echo -e "${BLUE}🛑 Stopping services...${NC}"
    docker compose -f "$COMPOSE_FILE" down
    echo -e "${GREEN}✅ Services stopped${NC}"
    echo ""
}

# Show logs
show_logs() {
    echo -e "${BLUE}📋 Service logs (press Ctrl+C to exit):${NC}"
    docker compose -f "$COMPOSE_FILE" logs -f
}

# Health check
check_health() {
    echo -e "${BLUE}🏥 Checking service health...${NC}"
    echo ""
    
    # Check OuRAGboros REST API
    echo -n "• REST API (port 8001): "
    if curl -s http://localhost:8001/docs > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Healthy${NC}"
    else
        echo -e "${RED}❌ Unhealthy${NC}"
    fi
    
    # Check OuRAGboros Logging API
    echo -n "• Logging API (port 8001): "
    if curl -s http://localhost:8001/logs/health > /dev/null 2>&1; then
        health_response=$(curl -s http://localhost:8001/logs/health)
        if echo "$health_response" | grep -q "healthy"; then
            echo -e "${GREEN}✅ Healthy${NC}"
        else
            echo -e "${YELLOW}⚠️ Partial (OpenSearch may be unavailable)${NC}"
        fi
    else
        echo -e "${RED}❌ Unhealthy (may need container rebuild)${NC}"
    fi
    
    # Check RAGAS Evaluator
    echo -n "• RAGAS Evaluator (port 8002): "
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Healthy${NC}"
    else
        echo -e "${RED}❌ Unhealthy${NC}"
    fi
    
    # Check OpenSearch
    echo -n "• OpenSearch (port 9200): "
    if curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Healthy${NC}"
    else
        echo -e "${RED}❌ Unhealthy${NC}"
    fi
    
    # Check Qdrant
    echo -n "• Qdrant (port 6333): "
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Healthy${NC}"
    else
        echo -e "${RED}❌ Unhealthy${NC}"
    fi
    
    # Check Ollama
    echo -n "• Ollama (port 11434): "
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Healthy${NC}"
    else
        echo -e "${RED}❌ Unhealthy${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}💡 Tips:${NC}"
    echo "  • If REST API is healthy but Logging API isn't, rebuild with: $0 restart"
    echo "  • If RAGAS Evaluator is unhealthy, check logs: $0 logs"
    echo "  • Open tools/query_logs_viewer.html in browser to view logged queries"
    echo ""
}

# Clean up
cleanup() {
    echo -e "${BLUE}🧹 Cleaning up...${NC}"
    
    # Stop services
    docker compose -f "$COMPOSE_FILE" down
    
    # Remove containers and unused images
    echo "Removing containers..."
    docker compose -f "$COMPOSE_FILE" rm -f
    
    echo "Removing unused images..."
    docker image prune -f
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
    echo ""
}

# Restart services (build + start)
restart_services() {
    echo -e "${BLUE}🔄 Restarting services...${NC}"
    stop_services
    build_containers
    start_services
}

# Parse arguments
NO_CACHE=false
COMMAND="build"

while [[ $# -gt 0 ]]; do
    case $1 in
        build)
            COMMAND="build"
            shift
            ;;
        build-all)
            COMMAND="build-all"
            shift
            ;;
        up)
            COMMAND="up"
            shift
            ;;
        down)
            COMMAND="down"
            shift
            ;;
        restart)
            COMMAND="restart"
            shift
            ;;
        logs)
            COMMAND="logs"
            shift
            ;;
        clean)
            COMMAND="clean"
            shift
            ;;
        health)
            COMMAND="health"
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main execution
echo -e "${GREEN}🐍 OuRAGboros Local Development${NC}"
echo "=================================="
echo ""

check_prerequisites

case $COMMAND in
    build)
        build_containers
        echo -e "${YELLOW}💡 Run '$0 up' to start services${NC}"
        ;;
    build-all)
        build_containers "all"
        echo -e "${YELLOW}💡 Run '$0 up' to start services${NC}"
        ;;
    up)
        start_services
        ;;
    down)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        show_logs
        ;;
    clean)
        cleanup
        ;;
    health)
        check_health
        ;;
    *)
        # Default: build and start
        build_containers
        start_services
        ;;
esac