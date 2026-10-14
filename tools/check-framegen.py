#!/usr/bin/env python3
"""PS5_FrameGen - the verdict of a FrameGen Lab run, from its log.

    tools/check-framegen.py LOG [--frames N] [--require METHOD:MIN_DB]... [--beat METHOD:BASE:MARGIN_DB]...
                                [--max-ms METHOD:MS]...

LOG is the run's log: klog/<run>/klog.log from the console (ps5/tools/run.sh) or
klog/host-<run>/log.txt from the PC (ps5/tools/host-reference.sh). The lab writes
one "framegen: frame ..." line per measured frame and one "framegen: summary ..."
line per method (examples/framegenlab/framegenlab.cpp).

It checks, and fails on any miss:
  - the lab started, and every method has a summary;
  - with --frames N (the run's budget): every odd frame the budget allows was
    measured, (N - 1) // 2 of them, for every method;
  - every PSNR is a finite number;
  - the scene moves, so a frame mixed from both neighbours is closer to the truth
    than one neighbour alone: blend beats repeat. If not, the measurement or the
    scene is broken, not a method;
  - --require METHOD:MIN_DB: the method's mean PSNR is at least MIN_DB;
  - --beat METHOD:BASE:MARGIN: its mean PSNR exceeds BASE's by MARGIN dB or more;
  - --max-ms METHOD:MS: its mean GPU time is at most MS (only where timed).

It writes LOG's folder/framegen.json (the summaries and checks) and ends with
"VERDICT PASS|FAIL: ...". Exit status 0 on PASS, 1 on FAIL, 2 on bad usage.

Copyright (C) 2026 Mihawk
SPDX-License-Identifier: MIT
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

LAB = re.compile(r"framegen: lab (\d+)x(\d+), (\d+) methods, gpu timing (on|off)")
FRAME_METHOD = re.compile(r" (\S+) (-?[\d.]+|inf|-inf|nan) dB (-?[\d.]+) ms")
SUMMARY = re.compile(r"framegen: summary (\S+) frames (\d+) psnr_mean (\S+) psnr_min (\S+) gpu_ms_mean (\S+)")


def pairs(values, parts, what):
    result = []
    for value in values or []:
        fields = value.split(":")
        if len(fields) != parts:
            sys.exit(f"{what} wants {parts} fields separated by ':', got {value!r}")
        result.append(fields)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("log", type=Path)
    parser.add_argument("--frames", type=int, help="the run's frame budget")
    parser.add_argument("--require", action="append", metavar="METHOD:MIN_DB")
    parser.add_argument("--beat", action="append", metavar="METHOD:BASE:MARGIN_DB")
    parser.add_argument("--max-ms", action="append", metavar="METHOD:MS")
    args = parser.parse_args()
    if not args.log.is_file():
        print(f"no log at {args.log}", file=sys.stderr)
        return 2
    text = args.log.read_text(errors="replace")

    lab = LAB.search(text)
    frames = {}  # method -> [(frame, psnr, ms)]
    for line in text.splitlines():
        if "framegen: frame " not in line:
            continue
        number = int(re.search(r"framegen: frame (\d+)", line).group(1))
        for method, psnr, ms in FRAME_METHOD.findall(line):
            frames.setdefault(method, []).append((number, float(psnr), float(ms)))
    summaries = {}
    for method, count, mean, low, gpu in SUMMARY.findall(text):
        summaries[method] = dict(frames=int(count), psnr_mean=float(mean), psnr_min=float(low), gpu_ms_mean=float(gpu))

    checks = []

    def check(ok, what):
        checks.append(dict(ok=bool(ok), check=what))

    check(lab, "the lab started (its 'framegen: lab' line)")
    methods = sorted(set(frames) | set(summaries))
    check(methods, "methods were measured")
    for method in methods:
        check(method in summaries, f"{method}: a summary line")
        measured = frames.get(method, [])
        if args.frames:
            expected = max(0, (args.frames - 1) // 2)
            check(len(measured) == expected, f"{method}: {len(measured)} frames measured, {expected} expected")
        check(all(math.isfinite(p) for _, p, _ in measured), f"{method}: every PSNR finite")
    mean = {m: s["psnr_mean"] for m, s in summaries.items()}
    if "blend" in mean and "repeat" in mean:
        check(mean["blend"] > mean["repeat"],
              f"sanity: blend ({mean['blend']:.2f} dB) beats repeat ({mean['repeat']:.2f} dB) on a moving scene")
    for method, minimum in pairs(args.require, 2, "--require"):
        got = mean.get(method)
        check(got is not None and got >= float(minimum), f"{method}: mean {got} dB >= {minimum} dB")
    for method, base, margin in pairs(args.beat, 3, "--beat"):
        got, ref = mean.get(method), mean.get(base)
        check(got is not None and ref is not None and got - ref >= float(margin),
              f"{method}: mean {got} dB beats {base} ({ref} dB) by {margin} dB")
    for method, limit in pairs(args.max_ms, 2, "--max-ms"):
        gpu = summaries.get(method, {}).get("gpu_ms_mean")
        if gpu is None or gpu < 0:
            check(False, f"{method}: GPU time measured (the run had no timing)")
        else:
            check(gpu <= float(limit), f"{method}: mean GPU time {gpu:.3f} ms <= {limit} ms")

    record = dict(log=str(args.log), lab=dict(zip(("width", "height", "methods", "timing"), lab.groups())) if lab else None,
                  summaries=summaries, checks=checks)
    out = args.log.parent / "framegen.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    for method in methods:
        s = summaries.get(method)
        if s:
            gpu = f"{s['gpu_ms_mean']:.3f} ms" if s["gpu_ms_mean"] >= 0 else "untimed"
            print(f"  {method:10s} {s['frames']:4d} frames  psnr mean {s['psnr_mean']:6.2f} dB  min {s['psnr_min']:6.2f} dB  gpu {gpu}")
    failed = [c["check"] for c in checks if not c["ok"]]
    for what in failed:
        print(f"  FAILED: {what}")
    size = f"{lab.group(1)}x{lab.group(2)}" if lab else "?"
    if failed:
        print(f"VERDICT FAIL: {len(failed)} of {len(checks)} checks failed ({size}); record {out}")
        return 1
    print(f"VERDICT PASS: {len(checks)} checks, {len(methods)} methods at {size}; record {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
