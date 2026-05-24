from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract numbered image frames from a video with ffmpeg."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, help="Optional output frames per second")
    parser.add_argument("--extension", default="jpg", choices=("jpg", "png"))
    parser.add_argument("--quality", default=2, type=int, help="JPEG quality for ffmpeg -q:v")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.video.is_file():
        raise RuntimeError(f"Video not found: {args.video}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    if args.output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"{args.output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = args.output_dir / f"%06d.{args.extension}"
    command = ["ffmpeg", "-hide_banner", "-y", "-i", str(args.video)]
    if args.fps:
        command.extend(["-vf", f"fps={args.fps}"])
    if args.extension == "jpg":
        command.extend(["-q:v", str(args.quality)])
    command.append(str(output_pattern))

    run(command)

    frame_count = sum(1 for _ in args.output_dir.glob(f"*.{args.extension}"))
    print(f"Extracted {frame_count} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
