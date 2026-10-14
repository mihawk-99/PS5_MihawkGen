#!/usr/bin/env bash
# PS5_FrameGen - the generator's host tests: seconds, no GPU, a verdict.
#
#   framegen/tests/run.sh
#
# Builds and runs test_framegen.cpp (the device sources linked to the PC's
# Vulkan loader, never reaching a device), and validates the library's SPIR-V
# with spirv-val. Ends with "VERDICT PASS|FAIL: ...".
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
set -uo pipefail
fg=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
root=$(dirname "$fg")
out="$root/build/framegen-tests"
mkdir -p "$out"
cxx=${CXX:-clang++}
# Vulkan's structures are filled with designated initializers that leave the rest zero, as the template does
flags=(-std=c++20 -O1 -g -Wall -Wextra -Werror -Wno-unknown-warning-option -Wno-missing-designated-field-initializers
       -I"$fg/include" -I"$fg/src" -I"$root/external")
failed=()

# The device sources link against the PC's Vulkan loader; the tests reach only
# the paths that refuse bad arguments before any Vulkan call
if "$cxx" "${flags[@]}" "$fg/tests/test_framegen.cpp" "$fg/src/fg_common.cpp" "$fg/src/fg_vulkan.cpp" "$fg/src/fg_neural.cpp" "$fg/src/fg_splat.cpp" -lvulkan \
        -o "$out/test_framegen" 2> "$out/build.log"; then
    "$out/test_framegen" "$fg/shaders" || failed+=("unit tests")
else
    cat "$out/build.log" >&2; failed+=("unit test build")
fi
spirv=0
for spv in "$fg/shaders/"*.spv; do
    spirv-val --target-env vulkan1.3 "$spv" || failed+=("spirv-val $(basename "$spv")")
    spirv=$((spirv + 1))
done
if (( ${#failed[@]} )); then
    echo "VERDICT FAIL: ${failed[*]}"
    exit 1
fi
echo "VERDICT PASS: unit tests (device-free paths), $spirv SPIR-V modules valid"
