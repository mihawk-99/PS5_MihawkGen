#!/usr/bin/env bash
# PS5_FrameGen - the RetroArch integration, end to end on this PC: a RetroArch built from the
# vendored source with the frame generation patch, run on a test core whose motion is known
# exactly, and the frames it generated scored against the scene at the half step.
#
#   integrations/retroarch/test-host.sh [NETWORK.fgw] [SUBFRAMES: 2 to 4, default "2 4"] [REPEAT: 1 to 4, default "1 2"]
#
# Needs the sibling PS5_RetroArch checkout's vendor/retroarch (PS5_RETROARCH_VENDOR overrides),
# a display, and a Vulkan device. Work goes to build/ra-host/ and build/ra-run/. Ends with
# "VERDICT PASS|FAIL" from check_dump.py.
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
vendor=${PS5_RETROARCH_VENDOR:-$root/../PS5_RetroArch/vendor/retroarch}
network=${1:-$root/framegen/models/network.fgw}
subframes=${2:-"2 4"}
repeats=${3:-"1 2"}
[[ -f $network ]] || { echo "no network at $network" >&2; exit 2; }
network=$(realpath "$network")
src=$root/build/ra-host/src
if [[ ! -x $src/retroarch ]]; then
    mkdir -p "$root/build/ra-host"
    rsync -a --exclude='.git' --exclude='*.o' "$vendor/" "$src/"
    (cd "$src" && patch -p1 < "$here/0001-vulkan-frame-generation.patch")
    cp "$here/gfx/vulkan_framegen.h" "$here/gfx/vulkan_framegen.cpp" "$src/gfx/drivers/"
    (cd "$src" && ./configure --disable-qt --disable-ffmpeg --disable-sdl --disable-sdl2 --disable-wayland --disable-pulse \
        --disable-pipewire --disable-oss --disable-v4l2 --disable-udev --disable-cdrom --disable-networking --enable-vulkan \
        --enable-x11 --disable-materialui --disable-ozone --enable-xmb --enable-rgui > ../configure.log 2>&1)
    sed -i "s|^LDFLAGS = .*|LDFLAGS = $root/build/libframegen_ra.a -lstdc++|" "$src/config.mk"
fi
"$here/build-host.sh" "$src"
(cd "$src" && rm -f retroarch && make -j"$(nproc)" > ../make.log 2>&1) || { tail -20 "$root/build/ra-host/make.log" >&2; exit 1; }
cc -O2 -shared -fPIC -I "$src/libretro-common/include" -o "$root/build/fg_test_core_libretro.so" "$here/fg_test_core.c" -lm
run=$root/build/ra-run
mkdir -p "$run"
status=0
for n in $subframes; do
for r in $repeats; do
    cat > "$run/test.cfg" <<CFG
video_driver = "vulkan"
video_fullscreen = "false"
video_vsync = "true"
audio_driver = "null"
audio_enable = "false"
input_driver = "x"
menu_driver = "rgui"
config_save_on_exit = "false"
video_shader_enable = "false"
video_framegen = "true"
video_shader_subframes = "$n"
video_swap_interval = "1"
CFG
    rm -rf "$run/dump" && mkdir "$run/dump"
    (cd "$run" && FG_TEST_REPEAT=$r PS5_FRAMEGEN_NETWORK=$network PS5_FRAMEGEN_DUMP=$run/dump timeout 120 "$src/retroarch" --config /dev/null \
        --appendconfig test.cfg -L "$root/build/fg_test_core_libretro.so" --max-frames=90 > "$run/retroarch.log" 2>&1) || true
    echo "== $n subframes, a new frame every $r core frame(s)"
    grep -E "FrameGen" "$run/retroarch.log" | head -3 || true
    python3 "$here/check_dump.py" "$root/build/fg_test_core_libretro.so" "$run/dump" --period "$r" || status=1
done
done
# A scene cut at core frame 40: the frames made across it must be the nearer real frame
cat > "$run/test.cfg" <<CFG
video_driver = "vulkan"
video_fullscreen = "false"
video_vsync = "true"
audio_driver = "null"
audio_enable = "false"
input_driver = "x"
menu_driver = "rgui"
config_save_on_exit = "false"
video_shader_enable = "false"
video_framegen = "true"
video_shader_subframes = "2"
video_swap_interval = "1"
CFG
rm -rf "$run/dump" && mkdir "$run/dump"
(cd "$run" && FG_TEST_CUT=40 PS5_FRAMEGEN_NETWORK=$network PS5_FRAMEGEN_DUMP=$run/dump timeout 120 "$src/retroarch" --config /dev/null \
    --appendconfig test.cfg -L "$root/build/fg_test_core_libretro.so" --max-frames=90 > "$run/retroarch.log" 2>&1) || true
echo "== a scene cut at core frame 40, 2 subframes"
python3 "$here/check_dump.py" "$root/build/fg_test_core_libretro.so" "$run/dump" --cut-after 40 || status=1
exit $status
