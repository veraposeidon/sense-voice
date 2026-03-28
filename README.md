# SenseVoice macOS Service

基于 `sensevoice-onnx` 的本地推理封装，提供 CLI 与 FastAPI HTTP 服务两种形态，方便研发团队在 macOS 上快速验证转写能力。

## 环境要求

- macOS（推荐 Apple Silicon）
- Python 3.9（建议使用 `uv` 管理虚拟环境）
- HuggingFace 访问权限（首次加载会自动下载 `lovemefan/SenseVoice-onnx` 资源，可通过 `HF_ENDPOINT` 配置镜像）

## 快速开始

```bash
# 1. 创建虚拟环境（固定 Python 3.9，避免 numpy 兼容问题）
uv venv --python 3.9 .venv

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 安装依赖（onnxruntime 包较大，必要时放宽下载超时）
UV_HTTP_TIMEOUT=120 uv pip install -r requirements.txt
```

首次运行推理时会自动将 SenseVoice 模型下载到 `sensevoice/resource/` 目录，可通过 `SenseVoiceConfig.download_path` 调整缓存位置。

## 项目结构

```
sensevoice-macos-service/
├── pyproject.toml                # 打包与 CLI entrypoint 定义
├── requirements.txt              # 运行依赖
├── scripts/                      # 开发辅助脚本
│   ├── run_cli.sh                # 激活 venv 后运行 CLI
│   └── run_server.sh             # 激活 venv 后运行 HTTP 服务
└── sv_service/
    ├── config.py                 # SenseVoiceConfig dataclass
    ├── engine.py                 # 单例推理引擎（封装模型、VAD、解析逻辑）
    ├── audio_io.py               # 音频归一化工具
    ├── schemas.py                # Pydantic 数据模型
    ├── cli.py                    # Typer CLI 实现
    └── http_app.py               # FastAPI 应用
```

## 使用方式

### CLI

等价地也可以使用脚本：`./scripts/run_cli.sh transcribe <path>`.

#### 常用命令速查

```bash
# 转写单个音频文件
sv transcribe path/to/audio.wav

# 转写目录下所有音频文件
sv transcribe path/to/audio/dir/

# 指定输出格式（默认 json）
sv transcribe path/to/audio.wav --output-format json

# 启用调试模式查看详细日志
sv transcribe path/to/audio.wav --debug
```

### HTTP 服务

```bash
source .venv/bin/activate
./scripts/run_server.sh  # 默认 127.0.0.1:8000
```

接口示例：

- `GET /healthz` → `{"status": "ok"}`
- `POST /v1/transcribe`，body:

```jsonc
{
  "audio_path": "/Users/me/audio/test.wav"
}
```

```shell
curl -X POST 127.0.0.1:8000/v1/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "/Users/shen/Downloads/hello.m4a"
  }'
```

返回字段包括 `text`、`segments`（含 start/end/language/emotion/event）以及 `raw.segments` 的原始标注。

## 注意事项

1. `sensevoice-onnx==1.1.0` 依赖 `setuptools<=65.0.0` 与旧版 `numpy`，务必使用独立 venv。
2. 如果下载 `onnxruntime` 超时，可设置 `UV_HTTP_TIMEOUT` 或提前手动下载 wheel。
3. 推理引擎 `SenseVoiceEngine` 为线程安全单例，可同时供 CLI 与 HTTP 使用。
4. 所有音频在进入推理前都会通过 `audio_io.normalize_audio_to_wav` 统一成 16k 单声道 wav。

完成 CLI/HTTP 验证后，可在同机起服务，给未来 SwiftUI/AppKit 客户端提供本地 API。如需了解整体流程与音频术语，可继续阅读 `docs/architecture.md`。

## 示例工作流

```bash
# 1. 准备音频文件
cp ~/Downloads/meeting.m4a ./test_audio/

# 2. 运行转写
sv transcribe test_audio/meeting.m4a

# 3. 查看结果（自动保存为 JSON）
cat test_audio/meeting.json
```

## 贡献指南

欢迎提交 Issue 和 PR！主要关注点：

- 新增功能请先开 Issue 讨论
- PR 需包含测试用例
- 代码风格遵循 PEP 8，使用 `black` 格式化
- 更新 README 和相关文档
