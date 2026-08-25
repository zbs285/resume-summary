from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .core import (
    ResumeSummaryError,
    default_output_dir,
    discover_pdfs,
    run_batch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resume-summary",
        description="批量简历事实抽取工具；PDF 文字在本机读取，默认调用在线 API。",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="PDF 文件或包含 PDF 的文件夹")
    parser.add_argument("--output", "-o", type=Path, help="输出文件夹")
    parser.add_argument(
        "--model",
        default="auto-free",
        help="默认 auto-free：实时选择当前免费模型；也可指定模型 ID",
    )
    parser.add_argument(
        "--provider",
        choices=("openrouter", "cpu", "ollama"),
        default="openrouter",
        help="默认 openrouter；cpu/ollama 仅作为离线备用",
    )
    parser.add_argument("--recursive", action="store_true", help="递归搜索子文件夹")
    parser.add_argument("--json-only", action="store_true", help="只生成 JSON，适合验证或自动化")
    parser.add_argument("--no-cache", action="store_true", help="忽略已有缓存并重新调用模型")
    parser.add_argument("--open", action="store_true", help="完成后在 Finder 中打开输出文件夹")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pdfs = discover_pdfs(args.inputs, recursive=args.recursive)
    output_dir = (args.output or default_output_dir(args.inputs[0])).expanduser().resolve()
    try:
        records = run_batch(
            pdfs,
            output_dir=output_dir,
            model=args.model,
            provider=args.provider,
            generate_reports=not args.json_only,
            use_cache=not args.no_cache,
        )
    except (ResumeSummaryError, KeyboardInterrupt) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    review_count = sum(bool(record["warnings"]) for record in records)
    print(f"完成：{len(records)} 份简历，{review_count} 份需复核")
    print(f"输出：{output_dir}")
    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(output_dir)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
