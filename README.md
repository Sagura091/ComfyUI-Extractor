# ComfyUI Document Extractor

A FastAPI microservice that processes documents and extracts images with captions, embedding text content into ChromaDB for semantic search.

## Features

- **Multi-format support**: PDF, PPTX, XLSX/XLSM, Markdown
- **Batch processing**: Handle ZIP/TAR archives or directory uploads
- **Image extraction**: Extract bitmaps with AI-generated captions
- **Semantic search**: Embed content into ChromaDB with sentence transformers
- **Output**: Generate PNG images with JSON metadata sidecars

## Project Structure

```
comfyui_2_0/
├── docker-compose.yaml     # Docker services configuration
├── tika-config.xml        # Apache Tika configuration
├── caption/               # Image captioning service
│   ├── Dockerfile
│   └── server.py
└── extractor/             # Main document processing service
    ├── Dockerfile
    ├── main.py           # FastAPI application
    ├── requirements.txt
    ├── settings.py       # Configuration settings
    ├── caption_api.py    # Caption service client
    ├── lora_export.py    # LoRA model export utilities
    ├── pdf_utils.py      # PDF processing
    ├── pptx_utils.py     # PowerPoint processing
    ├── xlsx_utils.py     # Excel processing
    └── md_utils.py       # Markdown processing
```

## Quick Start

### Using Docker Compose

1. Start the services:
   ```bash
   cd comfyui_2_0
   docker-compose up -d
   ```

2. The extractor service will be available at `http://localhost:8000`

### API Endpoints

- `POST /ingest` - Process a single file
- `POST /batch` - Process multiple files (ZIP, TAR, or directory upload)

### Example Usage

```bash
# Process a single PDF
curl -X POST "http://localhost:8000/ingest" \
     -F "file=@document.pdf"

# Process a ZIP archive
curl -X POST "http://localhost:8000/batch" \
     -F "file=@documents.zip"
```

## Configuration

Edit `extractor/settings.py` to configure:
- Input/output directories
- ChromaDB connection settings
- Embedding model selection
- Caption service endpoint

## Requirements

- Python 3.8+
- Docker & Docker Compose
- ChromaDB instance
- Caption service (for image captioning)

## Development

1. Install dependencies:
   ```bash
   cd comfyui_2_0/extractor
   pip install -r requirements.txt
   ```

2. Run the service locally:
   ```bash
   python main.py
   ```

## License

[Add your license here]
