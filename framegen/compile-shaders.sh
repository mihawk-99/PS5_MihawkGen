#!/usr/bin/env bash
# PS5_FrameGen - compile the generator's compute shaders into the library.
#
#   framegen/compile-shaders.sh
#
# framegen/shaders/<name>.comp -> framegen/shaders/<name>.comp.spv (for spirv-val)
# and framegen/src/shaders/<name>.comp.h (the same SPIR-V as a uint32_t array,
# fg_spv_<name>, which fg_vulkan.cpp builds its pipelines from: the library
# carries its shaders, and a title packages no files for it). Commit all three
# after changing a shader, as the template does with its SPIR-V.
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
command -v glslangValidator >/dev/null || { echo "glslangValidator was not found" >&2; exit 2; }
mkdir -p "$here/src/shaders"
for source in "$here/shaders/"*.comp; do
    name=$(basename "$source" .comp)
    glslangValidator -V --target-env vulkan1.3 "$source" -o "$source.spv" > /dev/null
    glslangValidator -V --target-env vulkan1.3 --vn "fg_spv_$name" "$source" -o "$here/src/shaders/$name.comp.h" > /dev/null
    echo "$name: $source.spv, src/shaders/$name.comp.h"
done
