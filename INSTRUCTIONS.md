# ComfyUI Document Extractor - Complete Instructions Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation & Setup](#installation--setup)
4. [Configuration](#configuration)
5. [Running the Service](#running-the-service)
6. [Using the API](#using-the-api)
7. [Supported File Formats](#supported-file-formats)
8. [Output Structure](#output-structure)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

## Overview

The ComfyUI Document Extractor is a powerful microservice that automatically:
- Extracts images from various document formats (PDF, PowerPoint, Excel, Markdown)
- Generates AI-powered captions for extracted images
- Embeds document content for semantic search using ChromaDB
- Outputs processed images with JSON metadata sidecars

## Prerequisites

Before you begin, ensure you have the following installed:

### Required Software
- **Docker Desktop** (recommended) or Docker Engine + Docker Compose
- **Git** (for cloning and version control)
- **Python 3.8+** (if running without Docker)

### System Requirements
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: At least 2GB free space for Docker images and processed files
- **Network**: Internet connection for downloading models and dependencies

## Installation & Setup

### Method 1: Using Docker (Recommended)

#### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd comfyui
```

#### Step 2: Start Services with Docker Compose
```bash
cd comfyui_2_0
docker-compose up -d
```

This will automatically:
- Pull required Docker images
- Start ChromaDB service
- Start the caption service
- Start the main extractor service
- Set up all necessary networking

#### Step 3: Verify Services are Running
```bash
docker-compose ps
```

You should see all services in "Up" status.

### Method 2: Manual Installation (Development)

#### Step 1: Set up ChromaDB
```bash
# Install ChromaDB server
pip install chromadb
# Start ChromaDB server
chroma run --host localhost --port 8000
```

#### Step 2: Set up Caption Service
```bash
cd comfyui_2_0/caption
pip install -r requirements.txt  # If requirements.txt exists
python server.py
```

#### Step 3: Set up Extractor Service
```bash
cd comfyui_2_0/extractor
pip install -r requirements.txt
python main.py
```

## Configuration

### Environment Variables

Create a `.env` file in the `comfyui_2_0/extractor/` directory:

```env
# Directories
RAW_DIR=/mnt/raw
OUT_DIR=/mnt/extract

# ChromaDB Configuration
CHROMA_HOST=localhost
CHROMA_PORT=8000

# AI Model Configuration
EMBED_MODEL=all-MiniLM-L6-v2

# Caption Service
CAPTION_SERVICE_URL=http://caption:5000
```

### Settings Configuration

Edit `comfyui_2_0/extractor/settings.py` to customize:

```python
# Directory paths
RAW_DIR = Path("/mnt/raw")
OUT_DIR = Path("/mnt/extract")

# ChromaDB connection
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

# Embedding model (sentence-transformers)
EMBED_MODEL = "all-MiniLM-L6-v2"

# Caption service endpoint
CAPTION_API_URL = "http://caption:5000/caption"
```

## Running the Service

### Using Docker Compose (Recommended)

```bash
# Start all services
cd comfyui_2_0
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Restart a specific service
docker-compose restart extractor
```

### Manual Startup

```bash
# Terminal 1: Start ChromaDB
chroma run --host localhost --port 8000

# Terminal 2: Start Caption Service
cd comfyui_2_0/caption
python server.py

# Terminal 3: Start Extractor Service
cd comfyui_2_0/extractor
python main.py
```

### Verifying the Service

Once running, verify the service is accessible:

```bash
curl http://localhost:8000/docs
```

This should show the FastAPI documentation interface.

## Using the API

### API Endpoints

The service exposes two main endpoints:

#### 1. Single File Processing: `/ingest`
Process individual documents one at a time.

#### 2. Batch Processing: `/batch`
Process multiple files from archives or directory uploads.

### Method 1: Using cURL

#### Process a Single PDF File
```bash
curl -X POST "http://localhost:8000/ingest" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/document.pdf"
```

#### Process a PowerPoint Presentation
```bash
curl -X POST "http://localhost:8000/ingest" \
     -F "file=@/path/to/your/presentation.pptx"
```

#### Process a ZIP Archive (Batch)
```bash
curl -X POST "http://localhost:8000/batch" \
     -F "file=@/path/to/your/documents.zip"
```

#### Process Excel Spreadsheet
```bash
curl -X POST "http://localhost:8000/ingest" \
     -F "file=@/path/to/your/spreadsheet.xlsx"
```

### Method 2: Using Python Requests

```python
import requests

# Single file processing
def process_single_file(file_path):
    url = "http://localhost:8000/ingest"
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    return response.json()

# Batch processing
def process_batch(archive_path):
    url = "http://localhost:8000/batch"
    with open(archive_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    return response.json()

# Example usage
result = process_single_file("document.pdf")
print(result)
```

### Method 3: Using the Web Interface

1. Open your browser and go to `http://localhost:8000/docs`
2. Click on the endpoint you want to use (`/ingest` or `/batch`)
3. Click "Try it out"
4. Upload your file using the file selector
5. Click "Execute"
6. View the response

## Supported File Formats

### PDF Files (.pdf)
- **What's extracted**: Images, charts, diagrams embedded in PDF
- **Text processing**: Full text content for semantic search
- **Output**: PNG images + JSON metadata for each page/image

### PowerPoint Presentations (.pptx)
- **What's extracted**: Slide images, embedded media, charts
- **Text processing**: Slide text, speaker notes, embedded text
- **Output**: PNG images for each slide + JSON metadata

### Excel Spreadsheets (.xlsx, .xlsm)
- **What's extracted**: Charts, embedded images, cell content visualizations
- **Text processing**: Cell values, formulas, sheet names
- **Output**: PNG images of charts/visuals + JSON metadata

### Markdown Files (.md)
- **What's extracted**: Embedded images, linked media
- **Text processing**: Full markdown content, headers, links
- **Output**: Processed images + JSON with structured content

## Output Structure

### Directory Layout
```
/mnt/extract/
├── document_name_uuid/
│   ├── page_001.png
│   ├── page_001.json
│   ├── page_002.png
│   ├── page_002.json
│   └── ...
```

### JSON Metadata Format
Each extracted image comes with a corresponding JSON file:

```json
{
  "doc_id": "document_uuid",
  "part_id": "page_001",
  "caption": "AI-generated description of the image content",
  "image_path": "relative/path/to/image.png",
  "raw_text": "Extracted text content (first 1000 chars)"
}
```

### ChromaDB Storage
All processed content is automatically stored in ChromaDB with:
- **Document embeddings**: Semantic vectors for search
- **Metadata**: Document ID, part ID, captions, file paths
- **Collection name**: `doc_chunks`

## Troubleshooting

### Common Issues

#### 1. Service Won't Start
**Problem**: Docker containers fail to start
**Solutions**:
```bash
# Check if ports are available
netstat -an | grep :8000

# Check Docker logs
docker-compose logs extractor

# Restart Docker Desktop
# Try different ports in docker-compose.yaml
```

#### 2. Out of Memory Errors
**Problem**: Large files cause memory issues
**Solutions**:
- Increase Docker memory allocation (Docker Desktop → Settings → Resources)
- Process smaller batches
- Use more specific file filtering

#### 3. ChromaDB Connection Failed
**Problem**: Cannot connect to ChromaDB
**Solutions**:
```bash
# Check if ChromaDB is running
docker-compose ps

# Restart ChromaDB service
docker-compose restart chromadb

# Check network connectivity
curl http://localhost:8000/api/v1/heartbeat
```

#### 4. Caption Service Unavailable
**Problem**: Image captioning fails
**Solutions**:
```bash
# Check caption service status
docker-compose logs caption

# Restart caption service
docker-compose restart caption

# Verify service is responding
curl http://localhost:5000/health
```

#### 5. File Processing Fails
**Problem**: Specific file types aren't processed
**Solutions**:
- Check file format is supported
- Verify file isn't corrupted
- Check file size limits
- Review error logs: `docker-compose logs extractor`

### Performance Optimization

#### For Large Files
```bash
# Increase processing timeout
# In docker-compose.yaml, add:
environment:
  - PROCESSING_TIMEOUT=300
```

#### For Batch Processing
```bash
# Optimize batch size
# Process in smaller chunks if memory limited
```

### Log Analysis

#### View Real-time Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f extractor

# Last 100 lines
docker-compose logs --tail=100 extractor
```

#### Debug Mode
Enable verbose logging by setting environment variable:
```bash
LOG_LEVEL=DEBUG
```

## Advanced Usage

### Custom Model Configuration

#### Change Embedding Model
Edit `settings.py`:
```python
# Use different sentence-transformer model
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Better quality
EMBED_MODEL = "sentence-transformers/all-MiniLM-L12-v2"  # Balanced
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # Faster
```

#### Custom Caption Models
Modify the caption service to use different models:
```python
# In caption/server.py
# Replace with your preferred captioning model
```

### Scaling and Production

#### Horizontal Scaling
```yaml
# In docker-compose.yaml
services:
  extractor:
    scale: 3  # Run 3 instances
```

#### Load Balancing
Add nginx or similar load balancer:
```yaml
services:
  nginx:
    image: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

#### Resource Limits
```yaml
services:
  extractor:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

### API Integration Examples

#### JavaScript/Node.js
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function processDocument(filePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    
    try {
        const response = await axios.post('http://localhost:8000/ingest', form, {
            headers: form.getHeaders()
        });
        return response.data;
    } catch (error) {
        console.error('Error processing document:', error);
    }
}
```

#### Python with AsyncIO
```python
import aiohttp
import asyncio

async def process_documents_async(file_paths):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for file_path in file_paths:
            task = process_single_async(session, file_path)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results

async def process_single_async(session, file_path):
    with open(file_path, 'rb') as f:
        data = aiohttp.FormData()
        data.add_field('file', f)
        
        async with session.post('http://localhost:8000/ingest', data=data) as response:
            return await response.json()
```

### Search and Query Examples

#### Search ChromaDB for Similar Content
```python
from chromadb import HttpClient

client = HttpClient(host="localhost", port=8000)
collection = client.get_collection("doc_chunks")

# Search for similar content
results = collection.query(
    query_texts=["financial charts and graphs"],
    n_results=10
)

print("Found documents:")
for result in results['documents']:
    print(f"- {result}")
```

#### Export Processed Data
```python
# Get all processed documents
all_docs = collection.get()

# Export to JSON
import json
with open('exported_docs.json', 'w') as f:
    json.dump(all_docs, f, indent=2)
```

## Support and Contributing

### Getting Help
- Check this instructions guide first
- Review the logs: `docker-compose logs`
- Open an issue on the GitHub repository
- Check the API documentation at `http://localhost:8000/docs`

### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Reporting Issues
When reporting issues, please include:
- Operating system and version
- Docker version
- Complete error logs
- Steps to reproduce
- Sample files (if safe to share)

---

**Need more help?** Check the main README.md or open an issue on GitHub!
