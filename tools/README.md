# OuRAGboros Tools

This directory contains utility tools for testing and debugging OuRAGboros.

## Contents

### query_logs_viewer.html
Interactive web interface for viewing and analyzing logged queries.
- View all logged RAG interactions 
- Search and filter by various fields
- Export data in JSON/CSV formats
- Real-time RAGAS evaluation scores

**Usage**: Open in browser and set API endpoint to `http://localhost:8001`

### streaming_test.html
Test tool for verifying the streaming API functionality.
- Tests Server-Sent Events (SSE) streaming
- Validates real-time token streaming
- Useful for debugging streaming issues

**Usage**: Open in browser and test against the streaming endpoints