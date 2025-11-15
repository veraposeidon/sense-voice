"""High-level SenseVoice inference wrapper shared by CLI and HTTP surfaces."""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download

from sensevoice.onnx.sense_voice_ort_session import SenseVoiceInferenceSession
from sensevoice.utils.frontend import WavFrontend
from sensevoice.utils.fsmn_vad import FSMNVad

from .config import DEFAULT_CONFIG, SenseVoiceConfig

LOGGER = logging.getLogger(__name__)

LANGUAGE_IDS = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
# SenseVoice 会在文本前插入 `<|NEUTRAL|><|Speech|>` 等标签，用集合定义便于快速判断
EMOTION_TAGS = {"NEUTRAL", "HAPPY", "SAD", "ANGRY", "CALM", "FEAR", "DISGUST"}
EVENT_TAGS = {
    "Speech",
    "Singing",
    "Music",
    "Laughter",
    "Applause",
    "Silence",
    "Noise",
}
TAG_PATTERN = re.compile(r"<\|([^|>]+)\|>")


class SenseVoiceEngine:
    """Singleton wrapper that keeps ONNX models resident in memory."""

    _instance_lock = threading.Lock()
    _instance: "SenseVoiceEngine | None" = None

    def __init__(self, cfg: SenseVoiceConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self._model_dir = Path(cfg.download_path).expanduser()
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model_dir = self._model_dir.resolve()
        self._language_id = LANGUAGE_IDS.get(cfg.language, LANGUAGE_IDS["auto"])
        self._run_lock = threading.Lock()
        self._ensure_resources()
        # WavFrontend 将波形转换为 SenseVoice 训练时所用的 fbank 特征
        self._frontend = WavFrontend(str(self._model_dir / "am.mvn"))
        # FSMNVad 用于截取“有声音”的片段，避免整体推理带来的长延迟
        self._vad = FSMNVad(str(self._model_dir))
        self._session = SenseVoiceInferenceSession(
            str(self._model_dir / "embedding.npy"),
            str(
                self._model_dir
                / ("sense-voice-encoder-int8.onnx" if cfg.use_int8 else "sense-voice-encoder.onnx")
            ),
            str(self._model_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model"),
            device_id=cfg.device,
            intra_op_num_threads=cfg.num_threads,
        )

    @classmethod
    def get_instance(cls, cfg: SenseVoiceConfig = DEFAULT_CONFIG) -> "SenseVoiceEngine":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(cfg)
        return cls._instance

    def transcribe_file(self, audio_path: str) -> Dict[str, Any]:
        """
        对单个音频执行推理，返回统一结构。
        `audio_path` 需指向 16k 单声道 wav（CLI/HTTP 已统一兑现）。
        """
        with self._run_lock:
            waveform, sample_rate = sf.read(audio_path, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)  # 兜底：若仍为多声道则取平均，保持与训练一致
            if sample_rate != 16000:
                raise ValueError("SenseVoiceEngine 仅接受 16k wav 输入，请先调用 audio_io.normalize_audio_to_wav")

            segments_ms = self._run_vad(waveform)
            processed_segments = self._decode_segments(waveform, segments_ms)

            if not processed_segments:
                processed_segments = self._decode_full_clip(waveform)

            text = " ".join(seg["text"] for seg in processed_segments).strip()
            language = processed_segments[0]["language"] if processed_segments else self.cfg.language
            emotion = processed_segments[0]["emotion"] if processed_segments else "NEUTRAL"
            event = processed_segments[0]["event"] if processed_segments else "Speech"

            raw_segments = [
                {
                    "start_ms": int(seg["start"] * 1000),
                    "end_ms": int(seg["end"] * 1000),
                    "text_raw": seg["raw_text"],
                }
                for seg in processed_segments
            ]

            return {
                "text": text,
                "language": language,
                "emotion": emotion,
                "event": event,
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"],
                        "language": seg["language"],
                        "emotion": seg["emotion"],
                        "event": seg["event"],
                    }
                    for seg in processed_segments
                ],
                "raw": {"segments": raw_segments},
            }

    def _ensure_resources(self) -> None:
        required = [
            "embedding.npy",
            "sense-voice-encoder.onnx",
            "sense-voice-encoder-int8.onnx",
            "am.mvn",
            "chn_jpn_yue_eng_ko_spectok.bpe.model",
        ]
        missing = [fname for fname in required if not (self._model_dir / fname).exists()]
        if missing:
            LOGGER.info("未找到完整模型资源，开始从 HuggingFace 下载（缺失: %s）", ", ".join(missing))
            snapshot_download(
                repo_id="lovemefan/SenseVoice-onnx",
                local_dir=str(self._model_dir),
                local_dir_use_symlinks=False,
            )

    def _run_vad(self, waveform: np.ndarray) -> Sequence[Sequence[int]]:
        # VAD 返回 (start_ms, end_ms) 列表，以毫秒为单位
        segments = self._vad.segments_offline(waveform)
        if not segments:
            duration_ms = int(len(waveform) / 16)
            return [(0, duration_ms)]
        return segments

    def _decode_segments(
        self, waveform: np.ndarray, segments_ms: Sequence[Sequence[int]]
    ) -> List[Dict[str, Any]]:
        decoded: List[Dict[str, Any]] = []
        for part in segments_ms:
            if len(part) < 2:
                continue
            start_ms, end_ms = int(part[0]), int(part[1])
            if end_ms <= start_ms:
                continue
            start_idx = max(0, start_ms * 16)
            end_idx = min(len(waveform), end_ms * 16)
            chunk = waveform[start_idx:end_idx]
            if chunk.size == 0:
                continue
            feats = self._frontend.get_features(chunk)  # → [帧数, 特征维度]
            if feats.size == 0:
                continue
            # SenseVoiceInferenceSession 本质是对 ONNX Runtime 的一次推理调用
            raw_text = self._session(
                feats[None, ...],
                language=self._language_id,
                use_itn=self.cfg.use_itn,
            )
            text, language, emotion, event = self._parse_output(raw_text)
            decoded.append(
                {
                    "start": start_ms / 1000.0,
                    "end": end_ms / 1000.0,
                    "text": text,
                    "language": language or self.cfg.language,
                    "emotion": emotion or "NEUTRAL",
                    "event": event or "Speech",
                    "raw_text": raw_text,
                }
            )
        return decoded

    def _decode_full_clip(self, waveform: np.ndarray) -> List[Dict[str, Any]]:
        # 当 VAD 未检测到片段时仍需对整段音频执行一次推理
        feats = self._frontend.get_features(waveform)
        raw_text = self._session(
            feats[None, ...],
            language=self._language_id,
            use_itn=self.cfg.use_itn,
        )
        text, language, emotion, event = self._parse_output(raw_text)
        return [
            {
                "start": 0.0,
                "end": len(waveform) / 16000.0,
                "text": text,
                "language": language or self.cfg.language,
                "emotion": emotion or "NEUTRAL",
                "event": event or "Speech",
                "raw_text": raw_text,
            }
        ]

    @staticmethod
    def _parse_output(raw_text: str) -> Tuple[str, str | None, str | None, str | None]:
        tags = TAG_PATTERN.findall(raw_text)  # e.g. ["zh", "NEUTRAL", "Speech"]
        clean_text = TAG_PATTERN.sub("", raw_text).strip()
        language = None
        emotion = None
        event = None
        for tag in tags:
            lowered = tag.lower()
            if lowered in LANGUAGE_IDS and language is None:
                language = tag
                continue
            upper = tag.upper()
            if upper in EMOTION_TAGS and emotion is None:
                emotion = upper
                continue
            if tag in EVENT_TAGS and event is None:
                event = tag
        return clean_text, language, emotion, event
