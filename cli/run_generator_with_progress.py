#!/usr/bin/env python3
import argparse
import queue
import subprocess
import sys
import threading
from pathlib import Path


def _model_args(generator_args):
    values = []
    for idx, arg in enumerate(generator_args):
        if arg == "--model" and idx + 1 < len(generator_args):
            values.append(generator_args[idx + 1])
    return values


def _is_onnx_run(generator_args):
    return any(str(value).endswith(".onnx") for value in _model_args(generator_args))


def _format_line(line):
    if line.startswith("DEBUG:piper.voice:"):
        return None
    for prefix in ("DEBUG:__main__:", "INFO:__main__:", "WARNING:__main__:", "ERROR:__main__:"):
        if line.startswith(prefix):
            return "   " + line[len(prefix):].strip()
    return line


def _reader(stdout, sink):
    try:
        for raw in stdout:
            sink.put(raw.rstrip("\n"))
    finally:
        sink.put(None)


def _progress_step(max_samples):
    if max_samples <= 20:
        return 1
    if max_samples <= 100:
        return 5
    return 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", required=True, type=int)
    parser.add_argument("generator_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    generator_args = list(args.generator_args)
    if generator_args and generator_args[0] == "--":
        generator_args = generator_args[1:]

    cmd = [sys.executable, args.generator, *generator_args]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    line_queue = queue.Queue()
    reader = threading.Thread(target=_reader, args=(proc.stdout, line_queue), daemon=True)
    reader.start()

    output_dir = Path(args.output_dir)
    use_sample_progress = _is_onnx_run(generator_args)
    step = _progress_step(args.max_samples)
    last_reported = 0
    stream_done = False

    while not stream_done or proc.poll() is None:
        try:
            line = line_queue.get(timeout=0.2)
        except queue.Empty:
            line = None

        if line is None:
            if not stream_done and not line_queue.empty():
                continue
            stream_done = proc.poll() is not None or stream_done
        else:
            formatted = _format_line(line)
            if formatted:
                print(formatted, flush=True)

        if use_sample_progress:
            current = len(list(output_dir.glob("*.wav")))
            should_report = current > last_reported and (
                current >= args.max_samples
                or current - last_reported >= step
            )
            if should_report:
                print(f"   Generated {current}/{args.max_samples} samples...", flush=True)
                last_reported = current

    rc = proc.wait()
    final_count = len(list(output_dir.glob("*.wav"))) if use_sample_progress else 0
    if use_sample_progress and final_count > last_reported:
        print(f"   Generated {final_count}/{args.max_samples} samples...", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
