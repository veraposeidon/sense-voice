"""Central configuration for the SenseVoice service."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SenseVoiceConfig:
    """Holds runtime settings shared by CLI and HTTP layers."""

    download_path: str = "sensevoice/resource"
    device: int = -1  # -1 means CPU in sensevoice-onnx
    num_threads: int = 4
    language: str = "auto"
    use_itn: bool = True
    use_int8: bool = True


DEFAULT_CONFIG = SenseVoiceConfig()
