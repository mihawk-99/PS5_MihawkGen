#!/usr/bin/env bash
# PS5_FrameGen - builds the generator as a static library for RetroArch's Vulkan driver, on
# this PC, and a RetroArch with the frame generation patch to try it with.
#
#   integrations/retroarch/build-host.sh [RETROARCH_SOURCE_TREE]
#
# The library is the generator (framegen/) and the driver's shim (gfx/vulkan_framegen.cpp),
# compiled with RetroArch's Vulkan symbol wrapper in front, so that every vk call goes
# through the pointers RetroArch loads. Output: build/libframegen_ra.a.
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
ra=${1:-$root/build/ra-host/src}
out=$root/build/libframegen_ra
mkdir -p "$out"
flags=(-std=c++20 -O2 -fPIC -Wall -Wno-missing-designated-field-initializers
       -include vulkan/vulkan_symbol_wrapper.h
       -I"$ra/libretro-common/include" -I"$ra" -I"$ra/gfx/include"
       -I"$root/framegen/include" -I"$root/framegen/src" -I"$here/gfx")
objects=()
for src in "$root/framegen/src/fg_common.cpp" "$root/framegen/src/fg_vulkan.cpp" "$root/framegen/src/fg_neural.cpp" "$root/framegen/src/fg_splat.cpp" \
           "$here/gfx/vulkan_framegen.cpp"; do
    obj="$out/$(basename "$src" .cpp).o"
    clang++ "${flags[@]}" -c "$src" -o "$obj"
    objects+=("$obj")
done
rm -f "$root/build/libframegen_ra.a"
ar rcs "$root/build/libframegen_ra.a" "${objects[@]}"
echo "built $root/build/libframegen_ra.a"
