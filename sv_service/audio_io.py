"""Audio normalization helpers shared by the CLI and HTTP layers."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Union

import librosa
import soundfile as sf

TARGET_SR = 16000
SUPPORTED_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


def normalize_audio_to_wav(src: Union[str, Path]) -> str:
    """
    将任意音频临时转换为 16k 单声道 wav，返回新文件路径。
    调用方负责在使用完毕后删除该临时文件。
    """
    path = Path(src).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"音频文件不存在: {path}")

    audio, sr = librosa.load(path.as_posix(), sr=None, mono=True)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    fd, tmp_path = tempfile.mkstemp(prefix="sv_audio_", suffix=".wav")
    os.close(fd)
    sf.write(tmp_path, audio, sr)
    return tmp_path


@contextmanager
def normalized_audio_file(src: Union[str, Path]) -> Generator[str, None, None]:
    """
    上下文管理器：返回转换后的 wav 路径，并在退出时自动清理。
    """
    tmp_path = normalize_audio_to_wav(src)
    try:
        yield tmp_path
    finally:
        Path(tmp_path).unlink(missing_ok=True)
