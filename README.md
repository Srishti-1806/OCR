# Form Extractor API
```mermaid
flowchart TD

A([Upload Form])

A --> B[Upload Agent]

B --> C{Input Type}

C -->|PDF| D[PDF Conversion Agent]
C -->|Image| E[Image Loader Agent]

D --> F[Image Preprocessing Agent]
E --> F

F --> G[OCR Agent<br/>PaddleOCR / Surya]

G --> H[OCR Parsing Agent]

H --> I[Layout Analysis Agent]

I --> J[Field Detection Agent]

J --> K[Confidence Evaluation Agent]

K --> L{Routing Decision}

L -->|High Confidence| M[Accept OCR Results]

L -->|Low Confidence / Handwritten| N[ROI Detection Agent]

N --> O[Crop Handwritten Region]

O --> P[Vision LLM Agent]

P --> Q[Extract Handwritten Text]

M --> R[Merge OCR + Vision Results]

Q --> R

R --> S[Consistency Validation Agent]

S --> T[Field Mapping Agent]

T --> U[Document Understanding Agent]

U --> V[Structured JSON Generator]

V --> W{Schema Validation}

W -->|Valid| X([Return JSON])

W -->|Invalid| Y[JSON Repair Agent]

Y --> W
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
