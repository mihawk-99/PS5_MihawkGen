#!/usr/bin/env python3
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
"""Same-device SDR byte parity against the pre-embedding revision, plus HDR readback.
Requires a Vulkan device, C++20 compiler, pkg-config vulkan and the baseline commit.
No downloaded fixtures; inputs include still detail, translation and a fading HUD.
"""
import hashlib
import io
import math
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "bb45cf1c536996e0bfc295ca91a2896e54d2c44f"
W, H = 64, 64


def checked(args, **kwargs):
    try:
        return subprocess.run([str(a) for a in args], check=True, **kwargs)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr)
        raise


def pq(nits):
    p = (nits / 10000) ** (2610 / 16384)
    return ((3424 / 4096 + (2413 / 128) * p) / (1 + (2392 / 128) * p)) ** (2523 / 32)


def main():
    env = dict(os.environ, FG_WORK_HEIGHT="64")
    for key in ("FG_CRISP", "FG_SOURCE", "FG_HDR10", "FG_INPUT_FORMAT", "FG_OUTPUT_FORMAT"):
        env.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="mihawk-formats-") as tmp:
        tmp = Path(tmp)
        baseline = tmp / "baseline"
        baseline.mkdir()
        archive = subprocess.check_output(["git", "archive", BASELINE, "framegen", "external/vulkan", "external/vk_video"], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(baseline, filter="data")
        flags = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "--libs", "vulkan"], text=True))
        compiler = shlex.split(os.environ.get("CXX", "clang++"))
        runner = ROOT / "framegen/tests/neural_run.cpp"
        for tree, name in ((baseline, "before"), (ROOT, "after")):
            src = tree / "framegen/src"
            command = compiler + ["-std=c++20", "-O2", "-I" + str(tree / "framegen/include"), "-I" + str(src), "-I" + str(tree / "external"), str(runner)]
            if name == "before":
                command += ["-DFG_LEGACY_CONTEXT"] + [str(src / f) for f in ("fg_common.cpp", "fg_vulkan.cpp", "fg_splat.cpp")]
            command += [str(src / "fg_neural.cpp")] + flags + ["-o", str(tmp / name)]
            checked(command)
        weights = ROOT / "framegen/models/network.fgw"

        def run(name, a, b, t, **options):
            (tmp / "a").write_bytes(a)
            (tmp / "b").write_bytes(b)
            result = checked([tmp / name, weights, tmp / "a", tmp / "b", W, H, t, tmp / "out"], env=dict(env, **options), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if not (tmp / "device").exists():
                print(result.stderr.decode().split("last run:")[0].strip())
                (tmp / "device").touch()
            return (tmp / "out").read_bytes()

        def sdr(shift=0, fade=0):
            image = bytearray()
            for y in range(H):
                for x in range(W):
                    xx = (x + shift) % W
                    value = ((xx // 5 + y // 7) % 2) * 160 + 30
                    rgb = (value, (xx * 5 + y * 3) % 256, (xx * 11 + y * 7) % 256)
                    if 6 < x < 32 and 5 < y < 14:
                        rgb = (220 - fade, 210 - fade, 200 - fade)
                    image.extend((*rgb, 255))
            return bytes(image)

        cases = ((sdr(), sdr()), (sdr(), sdr(3)), (sdr(), sdr(2, 40)))
        digest = hashlib.sha256()
        for a, b in cases:
            for t in (0.25, 0.5, 0.75):
                old, new = run("before", a, b, t), run("after", a, b, t)
                assert old == new, f"SDR bytes differ at t={t}"
                digest.update(new)
        print("PASS SDR: 9 outputs bit-identical to", BASELINE, "sha256=" + digest.hexdigest())

        def hdr(shift=0):
            values = []
            for y in range(H):
                for x in range(W):
                    xx = (x + shift) % W
                    nits = (0.01, 1, 20, 80, 203, 400, 1000, 4000)[(xx // 8 + y // 8) % 8]
                    rgb = [round(pq(nits * scale) * 1023) for scale in (1, 0.7, 0.3)]
                    if 8 < x < 36 and 5 < y < 15:
                        rgb = [round(pq(203) * 1023)] * 3
                    values.append(rgb[0] | (rgb[1] << 10) | (rgb[2] << 20) | (3 << 30))
            return struct.pack("<" + "I" * len(values), *values)

        a, b = hdr(), hdr(3)
        hdr_options = dict(FG_HDR10="1", FG_INPUT_FORMAT="rgb10_a2")
        still = run("after", a, a, 0.5, FG_OUTPUT_FORMAT="rgb10_a2", **hdr_options)
        assert still == a, "HDR10 still/HUD must retain all ten bits"
        half = run("after", a, a, 0.5, FG_OUTPUT_FORMAT="rgba16f", **hdr_options)
        half_values = struct.unpack("<" + "e" * (len(half) // 2), half)
        for pixel, packed in enumerate(struct.unpack("<" + "I" * (len(a) // 4), a)):
            for channel in range(3):
                expected = ((packed >> (10 * channel)) & 1023) / 1023
                assert abs(half_values[pixel * 4 + channel] - expected) <= 0.0005
            assert half_values[pixel * 4 + 3] == 1
        # RGBA16F input carries PQ too; it is not interpreted as linear scRGB.
        half_still = run("after", half, half, 0.5, FG_HDR10="1", FG_INPUT_FORMAT="rgba16f", FG_OUTPUT_FORMAT="rgba16f")
        assert half_still == half, "RGBA16F PQ still frame must be exact"
        moving = []
        for t in (0.25, 0.5, 0.75):
            out = run("after", a, b, t, FG_OUTPUT_FORMAT="rgba16f", **hdr_options)
            values = struct.unpack("<" + "e" * (len(out) // 2), out)
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
            assert max(values[0::4]) > pq(1000), "HDR highlight range lost"
            moving.append(out)
        assert len(set(moving)) == 3, "caller-scheduled interpolation times must change output"
        print("PASS HDR10: exact 10-bit still/HUD; A2B10G10R10 -> RGBA16F PQ; RGBA16F input; moving t=1/4,1/2,3/4 finite with highlights")
        print("VERDICT PASS: neural external-dispatch SDR parity and HDR formats (64-row test working size; no PS5 quality/performance claim)")


if __name__ == "__main__":
    main()
