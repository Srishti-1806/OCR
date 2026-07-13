import json
import sys
from pathlib import Path
from typing import List

from app.services.extraction_service import extract_fields
from app.services.ocr_service import run_full_ocr_pipeline


def process_pdf(pdf_path: Path) -> Path:
    pdf_path = pdf_path.resolve()
    output_path = pdf_path.with_suffix(".json")
    data = pdf_path.read_bytes()

    ocr_result = run_full_ocr_pipeline(data, pdf_path.name)
    extraction_result = extract_fields(ocr_result)

    payload = {
        "success": True,
        "data": extraction_result.dict(),
        "error": None,
    }

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def main() -> None:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
    pdf_files: List[Path] = sorted(root.rglob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {root}")
        return

    for pdf_file in pdf_files:
        output_path = process_pdf(pdf_file)
        print(f"Processed: {pdf_file.name} -> {output_path.name}")


if __name__ == "__main__":
    main()
