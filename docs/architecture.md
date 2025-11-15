# SenseVoice 本地服务架构说明

> 本文面向第一次接触音频推理的同学，重点解释整体流程与常见术语。若只想快速上手命令，可参考 `README.md`。

## 1. 项目整体逻辑

```
音频文件 -> 音频归一化(audio_io) -> SenseVoiceEngine (VAD 切片 + ONNX 推理)
          -> 统一结果结构 -> CLI / HTTP API / 未来 GUI
```

1. **CLI (`sv-cli transcribe`)**：更适合单次测试或批量转写目录。入口在 `sv_service/cli.py`。
2. **HTTP 服务 (`/v1/transcribe`)**：提供本地 FastAPI，方便后续 SwiftUI/AppKit 客户端或其他工具直接调用。
3. **SenseVoiceEngine**：真正负责加载模型、执行推理；CLI 和 HTTP 只是把请求参数转换成标准格式后调用它。

## 2. 关键模块

| 模块 | 作用 | 备注 |
| ---- | ---- | ---- |
| `sv_service/config.py` | 定义 `SenseVoiceConfig`，集中管理模型缓存路径、线程数、language、是否 int8 等开关 | 可在 CLI / HTTP 启动前统一配置 |
| `sv_service/audio_io.py` | 使用 `librosa`/`soundfile` 把任意格式音频转成 16k 单声道 WAV | SenseVoice 模型只接受 16k mono PCM，因此必须先归一化 |
| `sv_service/engine.py` | 懒加载模型，执行 VAD、提取特征、解析 SenseVoice 输出标签 | 是 CLI、HTTP 共享的核心，保证只加载一次模型节省内存 |
| `sv_service/cli.py` | Typer 命令行入口，负责遍历文件/目录并调用 Engine | 支持输出 JSON，适合脚本化 |
| `sv_service/http_app.py` | FastAPI 应用，提供 `GET /healthz` 和 `POST /v1/transcribe` | 未来 macOS GUI 可以直接请求此服务 |
| `sv_service/schemas.py` | 定义 Pydantic Request/Response/Segment 结构体 | 保证 HTTP 层有类型校验 |

## 3. 调用链详情

1. **音频归一化**：`audio_io.normalize_audio_to_wav` 将输入路径读入 `librosa`，如果采样率不为 16k 则重采样，并保证只有一个声道（mono）。
2. **Engine 初始化**：
   - `_ensure_resources`：如果本地 `sensevoice/resource` 缺少模型文件，就从 HuggingFace (`lovemefan/SenseVoice-onnx`) 拉取。`HF_ENDPOINT` 可重定向镜像。
   - `_frontend`：`WavFrontend` 用于把波形转成符合 SenseVoice 预期的特征。
   - `_vad`：`FSMNVad`（基于 FSMN 的 Voice Activity Detection）负责判断音频里有声段落，减少空白区推理耗时。
   - `_session`：封装 ONNX Runtime 推理。`use_int8` 选择 INT8 量化模型或 FP32 版本。
3. **推理流程** (`transcribe_file`)：
   - 读取 WAV → 若还有多声道则平均成单声道 → VAD 切出多个 (start_ms, end_ms) 段。
   - 对每段调用 `_frontend.get_features` 处理成 SenseVoice 需要的张量，丢给 `_session(...)`。
   - SenseVoice 输出文本串包含 `<|zh|><|NEUTRAL|><|Speech|>` 这种标签，`_parse_output` 使用正则拆出语言/情绪/事件，并去掉标签后的纯文本。
   - 最终返回一个统一 dict：`text`（整段拼接结果）、`segments`（每段的 start/end/text/language/emotion/event）、`raw`（保留部分原始信息方便调试）。

## 4. 音频术语快速解释

| 术语 | 含义 | 在项目里的位置 |
| ---- | ---- | ---- |
| **采样率 (Sample Rate)** | 每秒采集的样本数，单位 Hz。本项目固定 16000Hz（16k）。 | `audio_io.TARGET_SR = 16000`，转写前必须重采样。 |
| **单声道 (Mono)** | 只有一个声道的数据。多声道 (Stereo) 会被求平均。 | `audio_io.normalize_audio_to_wav` 中 `mono=True`。 |
| **VAD (Voice Activity Detection)** | 判断音频中哪些位置有声音，以减少空白片段的推理。 | `FSMNVad` 在 `engine._run_vad` 被调用。 |
| **特征提取 (Frontend / Features)** | 将原始波形转换为模型输入（如 fbank 特征）。 | `WavFrontend.get_features`。 |
| **ONNX Runtime** | 通用推理引擎，可在 CPU/GPU 上运行导出的模型。 | `SenseVoiceInferenceSession`、`VadOrtInferRuntimeSession` 都基于它。 |
| **INT8 量化** | 将模型参数压缩为 8-bit，加速推理但略微影响精度。 | `SenseVoiceConfig.use_int8` 控制是否使用 `sense-voice-encoder-int8.onnx`。 |

## 5. 如何扩展

- **改配置**：如果需要固定语言或禁用 ITN，可在 `SenseVoiceConfig` 调整，并在 CLI/HTTP 启动前传入新的配置。
- **GUI 客户端**：直接调用 FastAPI `/v1/transcribe` 即可；若需要上传音频二进制，可在 `schemas.TranscribeRequest` 增加 `base64` 字段，然后在 `http_app` 解码后调用 `audio_io`.
- **批处理/调度**：CLI 已支持目录扫描，可结合 `cron` 或其他任务系统定期运行，并把 `-o result.json` 输出给后续 pipeline。

## 6. 常见问题

1. **模型下载慢**：设置 `HF_ENDPOINT=https://hf-mirror.com` 或提前手动下载至 `sensevoice/resource/`。
2. **安装依赖冲突**：`sensevoice-onnx==1.1.0` 和 `onnxruntime==1.18.0` 依赖旧版 `numpy` 与 `setuptools`，必须使用独立 `.venv`。建议用 `uv venv --python 3.9 .venv`。
3. **内存占用高**：模型与缓存默认驻留内存；若仅测试 CLI，可在每次运行后让 Python 进程退出，以释放内存。

如需更深入的模型细节，可阅读 `sensevoice-onnx` 官方仓库的 `sensevoice/sense_voice.py` 等文件。本项目的 `engine.py` 已将核心逻辑封装成更易理解的形态。

