"""
Pydantic models shared across the OCR Service, Extraction Service, and API Service.
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def width(self) -> float:
        return self.x2 - self.x1


class OCRToken(BaseModel):
    """A single OCR-detected text block, normalized across engines."""
    text: str
    bbox: BoundingBox
    confidence: float
    page: int = 0
    line_id: Optional[int] = None


class OCRResult(BaseModel):
    """Full output of the OCR Service for one uploaded file."""
    filename: str
    page_count: int
    tokens: List[OCRToken]
    avg_confidence: float = 0.0


class ExtractedField(BaseModel):
    label: str
    value: Union[str, List[Union[str, Dict[str, Any]]], Dict[str, Any], None]
    confidence: float = 1.0
    field_type: str = "text"          # text | multiline | checkbox | table
    valid: bool = True
    validation_message: Optional[str] = None


class ExtractionResult(BaseModel):
    filename: str
    page_count: int
    fields: List[ExtractedField]
    raw_token_count: int = 0
    llm_attempts: int = 1
    warnings: List[str] = Field(default_factory=list)


class ExtractionResponse(BaseModel):
    success: bool
    data: Optional[ExtractionResult] = None
    error: Optional[str] = None
