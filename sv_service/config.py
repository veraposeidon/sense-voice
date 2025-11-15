"""Central configuration for the SenseVoice service."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SenseVoiceConfig:
    """Holds runtime settings shared by CLI and HTTP layers."""

    download_path: str = "sensevoice/resource"  # 模型缓存目录，首次运行会从 HF 下载
    device: int = -1  # -1 表示 CPU，未来如需 GPU 则传入具体卡号（需 onnxruntime-gpu）
    num_threads: int = 4  # ONNX Runtime 的线程数，影响 CPU 推理吞吐
    language: str = "auto"  # SenseVoice 支持 auto/zh/en/yue/ja/ko/nospeech
    use_itn: bool = True  # ITN（Inverse Text Normalization）用于数字、时间等格式化
    use_int8: bool = True  # 是否加载 INT8 量化模型，速度快但略有精度损失


DEFAULT_CONFIG = SenseVoiceConfig()
