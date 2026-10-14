#!/usr/bin/env bash
# PS5_FrameGen - runs a real core on this PC through build/ra-host2's RetroArch (integrations/retroarch/build-host2.sh)
# and the proxy core, which loads a save state and presses buttons by a script, and dumps the frames the frame
# generator made (check_dump-style files in OUTDIR: cur_N.rgba, gen_N_k.rgba, gen_N_k.t, size.txt).
#
#   run-host-game.sh CORE.so CONTENT OUTDIR [STATE] [INPUT "frame:BUTTON:frames,..."] [CORE_FRAMES=600] [SUBFRAMES=2] [SKIP=0] [COUNT=100]
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
core=$1; content=$2; out=$3; state=${4:-}; input=${5:-}; frames=${6:-600}; sub=${7:-2}; skip=${8:-0}; count=${9:-100}
rm -rf "$out" && mkdir -p "$out"
cat > "$out/test.cfg" <<CFG
video_driver = "vulkan"
video_fullscreen = "false"
video_vsync = "false"
pause_nonactive = "false"
audio_driver = "null"
audio_enable = "false"
menu_driver = "rgui"
config_save_on_exit = "false"
savestate_auto_save = "false"
savestate_auto_load = "false"
cheevos_enable = "false"
video_shader_enable = "false"
video_framegen = "true"
video_framegen_osd = "true"
video_framegen_auto = "${AUTO:-false}"
video_framegen_target = "${TARGET:-120}"
video_framegen_size = "${SIZE:-0}"
video_shader_subframes = "$sub"
video_refresh_rate = "119.88"
video_swap_interval = "1"
video_framegen_cadence = "${CADENCE:-0}"
video_framegen_method = "${METHOD:-0}"
system_directory = "$HOME/.config/retroarch/system"
CFG
(cd "$out" && FG_REAL_CORE=$core FG_STATE=$state FG_STATE_FRAME=5 FG_INPUT=$input PS5_FRAMEGEN_NETWORK=${NETWORK:-$root/framegen/models/network.fgw} \
    PS5_FRAMEGEN_DUMP=$out PS5_FRAMEGEN_DUMP_SKIP=$skip PS5_FRAMEGEN_DUMP_COUNT=$count \
    timeout -k 5 300 "$root/build/ra-host2/src/retroarch" --config /dev/null --appendconfig "$out/test.cfg" \
    -L "$root/build/fg_proxy_libretro.so" "$content" --max-frames="$frames" ${RA_EXTRA:-} > "$out/retroarch.log" 2>&1) || true
grep -E "fg_proxy|FrameGen\]" "$out/retroarch.log" | head -5
echo "dump: $(ls "$out" | grep -c '^gen_.*rgba') generated frames in $out"
