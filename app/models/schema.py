from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class FormField(BaseModel):
    label: str
    value: Union[str, List[Union[str, Dict[str, Any]]], Dict[str, Any], None]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = Field(default="paddle")


class ExtractionDocument(BaseModel):
    document_type: str = Field(default="unknown")
    fields: List[FormField] = Field(default_factory=list)


class ExtractionResponseModel(BaseModel):
    document_type: str = Field(default="unknown")
    fields: List[FormField] = Field(default_factory=list)


class OCRCandidate(BaseModel):
    text: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = Field(default="paddle")
    bbox: Optional[Dict[str, float]] = None


class OCRBundle(BaseModel):
    source: str
    candidates: List[OCRCandidate] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
