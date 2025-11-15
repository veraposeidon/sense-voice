"""FastAPI app exposing the local SenseVoice service."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .audio_io import normalized_audio_file
from .engine import SenseVoiceEngine
from .schemas import Segment, TranscribeRequest, TranscribeResponse

app = FastAPI(title="SenseVoice Local Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SenseVoiceEngine.get_instance()  # 服务启动时加载模型，后续请求共享


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/v1/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    if not req.audio_path:
        raise HTTPException(status_code=400, detail="audio_path is required")

    file_path = Path(req.audio_path).expanduser()
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"audio_path not found: {file_path}")

    # CLI 与 HTTP 共用统一入口：确保输入 wav 是 16k/mono
    with normalized_audio_file(file_path) as normalized:
        result = engine.transcribe_file(normalized)

    segments = [
        Segment(
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            text=seg.get("text", ""),
            language=seg.get("language", "auto"),
            emotion=seg.get("emotion", "NEUTRAL"),
            event=seg.get("event", "Speech"),
        )
        for seg in result.get("segments", [])
    ]

    return TranscribeResponse(
        text=result.get("text", ""),
        language=result.get("language", "auto"),
        emotion=result.get("emotion", "NEUTRAL"),
        event=result.get("event", "Speech"),
        segments=segments,
        raw=result.get("raw", {}),
    )
