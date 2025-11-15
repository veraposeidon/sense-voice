"""Typer-based CLI entry for local transcription."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import typer

from .audio_io import SUPPORTED_SUFFIXES, normalized_audio_file
from .engine import SenseVoiceEngine

app = typer.Typer(help="SenseVoice 本地 CLI，支持单文件与目录批量转写。")


@app.command()
def transcribe(
    input: str = typer.Argument(..., help="音频文件或目录路径"),
    output_json: Optional[str] = typer.Option(
        None, "--output-json", "-o", help="将结果写入 JSON 文件"
    ),
) -> None:
    engine = SenseVoiceEngine.get_instance()
    input_path = Path(input).expanduser()
    if not input_path.exists():
        typer.secho(f"路径不存在: {input_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    audio_files = list(_iter_audio_files(input_path))
    if not audio_files:
        typer.secho("未找到可用音频文件。", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    results: List[dict] = []
    for audio_file in audio_files:
        try:
            # normalized_audio_file 会在 with 块结束后移除临时 wav，避免磁盘垃圾
            with normalized_audio_file(audio_file) as normalized:
                res = engine.transcribe_file(normalized)
            results.append({"file": str(audio_file), **res})
            typer.secho(f"[OK] {audio_file}", fg=typer.colors.GREEN)
        except Exception as exc:  # pragma: no cover - CLI 交互输出
            typer.secho(f"[FAIL] {audio_file}: {exc}", fg=typer.colors.RED)

    if output_json:
        Path(output_json).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        typer.secho(f"结果已写入 {output_json}", fg=typer.colors.BLUE)
    else:
        for item in results:
            typer.echo(f"=== {item['file']} ===")
            typer.echo(item["text"])
            typer.echo("")


def _iter_audio_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path
        return

    for candidate in path.rglob("*"):
        if candidate.suffix.lower() in SUPPORTED_SUFFIXES:
            yield candidate


if __name__ == "__main__":
    app()
