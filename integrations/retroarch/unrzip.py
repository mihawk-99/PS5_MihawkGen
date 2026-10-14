#!/usr/bin/env python3
"""PS5_FrameGen - RetroArch's compressed save states (RZIP) as raw core states: unrzip.py IN OUT.

Header: "#RZIPv", version, "#", chunk size (u32), total size (u64); then, per chunk, its compressed size (u32) and zlib data.
Copyright (C) 2026 Mihawk. SPDX-License-Identifier: GPL-3.0-or-later"""
import struct, sys, zlib
data = open(sys.argv[1], "rb").read()
if data[:6] != b"#RZIPv":
    open(sys.argv[2], "wb").write(data); sys.exit(0)
chunk, total = struct.unpack_from("<IQ", data, 8)
pos, out = 20, bytearray()
while pos < len(data) and len(out) < total:
    (size,) = struct.unpack_from("<I", data, pos); pos += 4
    out += zlib.decompress(data[pos:pos + size]); pos += size
state = bytes(out[:total])
if state[:7] == b"RASTATE":            # RetroArch's wrapper: blocks of 4-letter id, u32 size, data padded to 8; "MEM " is the core's
    pos = 8
    while pos + 8 <= len(state):
        ident, size = state[pos:pos + 4], struct.unpack_from("<I", state, pos + 4)[0]
        pos += 8
        if ident == b"MEM ":
            state = state[pos:pos + size]
            break
        pos += (size + 7) & ~7
open(sys.argv[2], "wb").write(state)
