#!/usr/bin/env bash
# PS5_FrameGen - a RetroArch for this PC with the same driver hooks the PS5 title has (patches 0119 to 0121 of
# ../PS5_RetroArch, applied to a copy of its vendored source), for runs on real cores before the console.
#
#   integrations/retroarch/build-host2.sh        -> build/ra-host2/src/retroarch
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
port=${PS5_RETROARCH:-$root/../PS5_RetroArch}
src=$root/build/ra-host2/src
mkdir -p "$src"
if [[ ! -f $src/.patched ]]; then
    rsync -a --exclude='.git' --exclude='*.o' "$port/vendor/retroarch/" "$src/"
    python3 - "$port/tools/apply-port-patches.py" "$src" <<'PY'
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("a", sys.argv[1]); a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
files = ("gfx/drivers/vulkan.c", "configuration.h", "configuration.c", "retroarch.c")
a.EDITS = [e for e in a.EDITS if e[0] in files and any(k in e[3] or k in e[2] for k in ("0119", "0120", "0121"))]
a.WITHDRAWN = set()
sys.argv = ["apply", sys.argv[2]]
raise SystemExit(a.main())
PY
    # the port's texture struct has a physical width; upstream's has not
    sed -i '/input.physical_width = vk->framegen_frame.width;/d' "$src/gfx/drivers/vulkan.c"
    cp "$here/gfx/vulkan_framegen.h" "$here/gfx/vulkan_framegen.cpp" "$src/gfx/drivers/"
    (cd "$src" && ./configure --disable-qt --disable-ffmpeg --disable-sdl --disable-sdl2 --disable-wayland --disable-pulse \
        --disable-pipewire --disable-oss --disable-v4l2 --disable-udev --disable-cdrom --disable-networking --enable-vulkan \
        --enable-x11 --disable-materialui --disable-ozone --enable-xmb --enable-rgui > ../configure.log 2>&1)
    sed -i "s|^LDFLAGS = .*|LDFLAGS = $root/build/libframegen_ra.a -lstdc++|" "$src/config.mk"
    touch "$src/.patched"
fi
"$here/build-host.sh" "$src"
cc -O2 -c -fPIC -o "$root/build/libframegen_ra/host_stubs.o" "$here/host_stubs.c"
ar rcs "$root/build/libframegen_ra.a" "$root/build/libframegen_ra/host_stubs.o"
(cd "$src" && rm -f retroarch && make -j"$(nproc)" > ../make.log 2>&1) || { tail -20 "$root/build/ra-host2/make.log" >&2; exit 1; }
echo "built $src/retroarch"
