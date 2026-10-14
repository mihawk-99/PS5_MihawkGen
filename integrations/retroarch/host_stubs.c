/* PS5_FrameGen - what the PS5 title's menu code and display setup give the driver hooks, for a RetroArch built on a PC
 * (integrations/retroarch/build-host2.sh): the size limit, the status display and the core rate.
 *
 * Copyright (C) 2026 Mihawk
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <stdbool.h>
#include <stdio.h>

#include <stdlib.h>

unsigned vulkan_ps5_display_refresh_x1000;
/* presents of each picture (Target Framerate): FG_DUP in the environment, for runs on a PC */
unsigned ps5_fg_dup = 1;
__attribute__((constructor)) static void ps5_fg_dup_init(void)
{
   const char *e = getenv("FG_DUP");
   if (e && atoi(e) >= 1)
      ps5_fg_dup = (unsigned)atoi(e);
}
float ps5_fg_core_fps;
float ps5_fg_dbg_t = 1.0f;
float ps5_fg_target_fps = 119.88f;   /* FG_TARGET in the environment overrides it, for runs on a PC */
__attribute__((constructor)) static void ps5_fg_target_init(void)
{
   const char *e = getenv("FG_TARGET");
   if (e && atof(e) > 1.0)
      ps5_fg_target_fps = (float)atof(e);
}

unsigned long ps5_fg_size_pixels_of(unsigned size)
{
   static const unsigned long px[] = { 3840ul * 2160ul, 2560ul * 1440ul, 1920ul * 1080ul, 1280ul * 720ul };
   return px[size < 4 ? size : 0];
}

int ps5_fg_floor;

void ps5_fg_osd(bool plan, unsigned subframes, unsigned period, unsigned width, unsigned height, unsigned long generated, unsigned long unique, unsigned level, double gen_ms, unsigned long calls)
{
   static unsigned long presents_seen;
   if (++presents_seen % 200 == 0)
      fprintf(stderr, "[FrameGen] host: presents %lu, generated %lu, new %lu, calls %lu, plan %d, subframes %u, period %u, %ux%u, level %u, %.2f ms\n", presents_seen, generated, unique, calls, plan, subframes, period, width, height, level, gen_ms);
}
