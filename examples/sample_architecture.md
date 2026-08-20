# OmniFind Architecture Overview

OmniFind is an AI-powered, context-aware retrieval system for heterogeneous digital assets.

## Pipeline Architecture

- File Discovery & Classification
- Format-Specific Content Extraction
- Semantic Intelligence Layer
- Local Vector & Metadata Indexing

### Key Services

| Service | Port | Protocol |
| :--- | :--- | :--- |
| Core API | 8000 | HTTP/REST |
| Web UI | 5173 | HTTP |
| Qdrant Vector DB | Local | Embedded |

*Note: All processing runs locally with zero mandatory cloud API costs.*
