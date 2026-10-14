#!/usr/bin/env bash
# PS5_FrameGen - the generator and the RetroArch shim compile with the PS5 toolchain (the pinned
# payload SDK's clang, through PS5_Vulkan's wrapper), against RetroArch's own headers and its
# Vulkan symbol wrapper: what the PS5 RetroArch title's build will do with them. Compile only.
#
#   integrations/retroarch/check-ps5-compile.sh
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
vendor=${PS5_RETROARCH_VENDOR:-$root/../PS5_RetroArch/vendor/retroarch}
sdk=${PS5_PAYLOAD_SDK:-$root/.deps/native/ps5-payload-sdk}
cc=${PS5_CLANG_WRAPPER:-$root/../PS5_Vulkan/tooling/prospero-clang18}
out=$root/build/ps5-shim
mkdir -p "$out"
for src in framegen/src/fg_common.cpp framegen/src/fg_vulkan.cpp framegen/src/fg_neural.cpp framegen/src/fg_splat.cpp integrations/retroarch/gfx/vulkan_framegen.cpp; do
    PS5_PAYLOAD_SDK="$sdk" sh "$cc" -x c++ -std=c++20 -O2 -fPIC -Wall -Werror -Wno-missing-designated-field-initializers \
        -include vulkan/vulkan_symbol_wrapper.h -I"$vendor/libretro-common/include" -I"$vendor" -I"$vendor/gfx/include" \
        -I"$root/framegen/include" -I"$root/framegen/src" -I"$here/gfx" -c "$root/$src" -o "$out/$(basename "$src" .cpp).o"
    echo "compiled $src"
done
echo "VERDICT PASS: 4 units compile for the PS5"
