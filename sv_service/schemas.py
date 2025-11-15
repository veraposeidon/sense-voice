"""Pydantic models for the HTTP API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TranscribeRequest(BaseModel):
    audio_path: Optional[str] = Field(
        default=None, description="本机可读的音频文件路径"
    )


class Segment(BaseModel):
    start: float
    end: float
    text: str
    language: str
    emotion: str
    event: str


class TranscribeResponse(BaseModel):
    text: str
    language: str
    emotion: str
    event: str
    segments: List[Segment] = []
    raw: Dict[str, Any] = {}
