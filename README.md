# Form Extractor API

```mermaid
flowchart TD
    A([Upload PDF or image]) --> B[Save file to upload dir]
    B --> C{Input type}
    C -->|PDF| D[Render pages with Poppler or PyMuPDF]
    C -->|Image| E[Load image directly]
    D --> F[Preprocess page images]
    E --> F
    F --> G[Run primary OCR]
    G --> H{OCR returned text?}
    H -->|No| I[Fallback to docTR]
    H -->|Yes| J{Confidence low?}
    I --> K{docTR returned text?}
    K -->|No| L[Fallback to Surya]
    K -->|Yes| J
    J -->|Yes| L
    J -->|No| M[Build OCR context]
    L --> M
    M --> N[LLM-based extraction]
    N --> O{Valid JSON?}
    O -->|No| P[Use fallback payload]
    O -->|Yes| Q[Return extraction result]
    P --> Q
```

with llm
```mermaid
flowchart TD

    Start([START])

    Start --> UploadAgent

    UploadAgent --> LoaderAgent

    LoaderAgent --> PreprocessingAgent

    PreprocessingAgent --> PaddleOCRAgent

    PaddleOCRAgent --> ConfidenceAgent

    ConfidenceAgent -->|High Confidence| MergeAgent

    ConfidenceAgent -->|Low Confidence| SuryaOCRAgent

    SuryaOCRAgent --> SuryaConfidenceAgent

    SuryaConfidenceAgent -->|High Confidence| MergeAgent

    SuryaConfidenceAgent -->|Low Confidence| ROICropAgent

    ROICropAgent --> VisionLLMAgent

    VisionLLMAgent --> MergeAgent

    MergeAgent --> ConsistencyValidationAgent

    ConsistencyValidationAgent --> FieldMappingAgent

    FieldMappingAgent --> DocumentUnderstandingAgent

    DocumentUnderstandingAgent --> JSONGeneratorAgent

    JSONGeneratorAgent --> SchemaValidationAgent

    SchemaValidationAgent -->|Valid| End([END])

    SchemaValidationAgent -->|Invalid| JSONRepairAgent

    JSONRepairAgent --> SchemaValidationAgent
```

**Why the split matters:** swapping PaddleOCR for another engine (Surya, Tesseract,
a cloud OCR API) only touches `ocr_service.py`. Swapping the LLM/provider only
touches `extraction_service.py`. The router never changes.

## Project layout

```
app/
  main.py                     FastAPI app entrypoint
  routers/
    extract.py                POST /extract orchestration
  services/
    ocr_service.py             OCR Service
    extraction_service.py      Extraction Service (LLM)
  models/
    schemas.py                 Pydantic models (OCRToken, ExtractedField, etc.)
  utils/
    geometry.py                 line-grouping / bbox helpers used by OCR service
    regex.py                    optional field-type regex, if you add rule-based validation later
  core/
    config.py                   env-driven settings
    logging_config.py           logging setup
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# system dependency for pdf2image
# Linux / WSL:
# sudo apt-get install poppler-utils
# Windows:
# install Poppler and add its Library\bin folder to PATH
# For example, with Chocolatey:
# choco install poppler -y

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY

uvicorn app.main:app --reload --port 8000
```

## Usage

```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@/path/to/form.pdf"
```

Response shape:

```json
{
  "success": true,
  "data": {
    "filename": "form.pdf",
    "page_count": 1,
    "fields": [
      {"label": "name", "value": "Rahul Sharma", "confidence": 0.94, "field_type": "text"},
      {"label": "date_of_birth", "value": "1998-04-12", "confidence": 0.91, "field_type": "text"},
      {"label": "terms_accepted", "value": true, "confidence": 0.88, "field_type": "checkbox"}
    ],
    "raw_token_count": 42,
    "llm_attempts": 1,
    "warnings": []
  },
  "error": null
}
```

## Notes / next steps

- `LLM_MAX_RETRIES` controls how many times the Extraction Service retries if
  the model returns invalid/malformed JSON.
- Table fields come back as `{"headers": [...], "rows": [[...], ...]}` inside `value`.
- If you want a rule-based fallback (no LLM available), `utils/regex.py` already
  has label keywords + validators ready to wire into a fallback path in
  `extraction_service.py`.
