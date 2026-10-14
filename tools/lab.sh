#!/usr/bin/env bash
# PS5_FrameGen - the loop, cheapest step first; each ends in a verdict.
#
#   tools/lab.sh tests              seconds: the generator's host tests
#   tools/lab.sh host [FRAMES]      a minute: FrameGen Lab on this PC's Vulkan driver, measured
#   tools/lab.sh console [FRAMES]   a few minutes: build, deploy, run on the console, measured
#   tools/lab.sh pixels             seconds: the pixel-art motion pass on a made-up scroll (needs a Vulkan device)
#   tools/lab.sh model [NETWORK]    a minute: the trained network (framegen/models/network.fgw, or a file) in
#                                   FrameGen Lab on the PC: it must beat blend by 8 dB and reach 38 dB
#
# FRAMES is the run's budget (300 by default, the reference pictures' too). Extra
# checks for tools/check-framegen.py go in CHECKS, for instance
#   CHECKS="--beat warp:blend:1 --max-ms warp:2" tools/lab.sh console
# The console is the one PS5_Vulkan's .env names, unless PS5_HOST is set in
# the environment for this command (never written to a file in this repository).
# The PC's timings say nothing about the console's (another GPU and driver):
# judge speed only on the console.
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: MIT
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
step=${1:-}
frames=${2:-300}
read -r -a checks <<< "${CHECKS:-}"

case $step in
tests)
    "$root/framegen/tests/run.sh"
    ;;
host)
    FRAMES=$frames bash "$root/ps5/tools/host-reference.sh" framegenlab
    run=$(ls -td "$root"/klog/host-* | head -1)
    python3 "$root/tools/check-framegen.py" "$run/log.txt" --frames "$frames" "${checks[@]}"
    ;;
console)
    bash "$root/ps5/tools/build.sh"
    bash "$root/ps5/tools/deploy.sh"
    FRAMES=$frames bash "$root/ps5/tools/run.sh" framegenlab
    run=$(ls -td "$root"/klog/2* | head -1)
    python3 "$root/tools/check-framegen.py" "$run/klog.log" --frames "$frames" "${checks[@]}"
    ;;
pixels)
    mkdir -p "$root/build"
    g++ -O2 -std=c++17 -I"$root/framegen/include" -I"$root/framegen/src" "$root/framegen/tests/neural_run.cpp" \
        "$root/framegen/src/fg_common.cpp" "$root/framegen/src/fg_vulkan.cpp" "$root/framegen/src/fg_neural.cpp" "$root/framegen/src/fg_splat.cpp" -lvulkan -o "$root/build/neural_run"
    python3 "$root/framegen/tests/check_pixel_motion.py" "$root/build/neural_run" "$root/framegen/models/network.fgw"
    ;;
model)
    network=$(realpath "${2:-$root/framegen/models/network.fgw}")
    FG_NETWORK=$network FRAMES=300 bash "$root/ps5/tools/host-reference.sh" framegenlab
    run=$(ls -td "$root"/klog/host-* | head -1)
    python3 "$root/tools/check-framegen.py" "$run/log.txt" --frames 300 --require neural:38 --beat neural:blend:8 "${checks[@]}"
    ;;
*)
    sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
