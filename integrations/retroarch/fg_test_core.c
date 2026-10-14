/*
 * PS5_FrameGen - a libretro core that only draws motion with an exactly known position at
 * every frame: a scrolling checkerboard under moving shapes and a fixed HUD, in 640x360
 * XRGB8888 at 60 fps (a content frame every 1/60 s, its motion a function of the frame
 * number alone). It is the content the frame generation is checked on in RetroArch where no
 * game may be used: a frame between two others is the same function at the half step.
 *
 *   cc -O2 -shared -fPIC -o fg_test_core_libretro.so integrations/retroarch/fg_test_core.c -lm
 *
 * Copyright (C) 2026 Mihawk
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "libretro.h"

#define W 640
#define H 360

static retro_video_refresh_t video_cb;
static retro_environment_t env_cb;
static retro_input_poll_t input_poll_cb;
static retro_input_state_t input_state_cb;
static retro_audio_sample_t audio_cb;
static retro_audio_sample_batch_t audio_batch_cb;
static uint32_t frame_buffer[W * H];
static unsigned frame_number;

/* FG_TEST_FPS (24 to 120, default 60): the core's frame rate. The scene's speed is the same in pictures a second whatever
 * it is (a step of 60/fps scene frames for each core frame), so the same motion is sampled at a different rate. */
static double core_fps(void)
{
	const char *s = getenv("FG_TEST_FPS");
	const double fps = s ? atof(s) : 60.0;
	return fps >= 24.0 && fps <= 120.0 ? fps : 60.0;
}

void retro_set_environment(retro_environment_t cb)
{
	env_cb = cb;
	bool yes = true;
	cb(RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME, &yes);
}
void retro_set_video_refresh(retro_video_refresh_t cb) { video_cb = cb; }
void retro_set_audio_sample(retro_audio_sample_t cb) { audio_cb = cb; }
void retro_set_audio_sample_batch(retro_audio_sample_batch_t cb) { audio_batch_cb = cb; }
void retro_set_input_poll(retro_input_poll_t cb) { input_poll_cb = cb; }
void retro_set_input_state(retro_input_state_t cb) { input_state_cb = cb; }
void retro_init(void) {}
void retro_deinit(void) {}
unsigned retro_api_version(void) { return RETRO_API_VERSION; }
void retro_get_system_info(struct retro_system_info *info)
{
	memset(info, 0, sizeof(*info));
	info->library_name = "FrameGen test core";
	info->library_version = "1";
	info->need_fullpath = false;
	info->valid_extensions = "";
}
void retro_get_system_av_info(struct retro_system_av_info *info)
{
	info->geometry.base_width = W;
	info->geometry.base_height = H;
	info->geometry.max_width = W;
	info->geometry.max_height = H;
	info->geometry.aspect_ratio = (float)W / H;
	info->timing.fps = core_fps();
	info->timing.sample_rate = 48000.0;
}
void retro_set_controller_port_device(unsigned port, unsigned device) { (void)port; (void)device; }
void retro_reset(void) { frame_number = 0; }
size_t retro_serialize_size(void) { return sizeof(frame_number); }
bool retro_serialize(void *data, size_t size)
{
	if (size < sizeof(frame_number)) return false;
	memcpy(data, &frame_number, sizeof(frame_number));
	return true;
}
bool retro_unserialize(const void *data, size_t size)
{
	if (size < sizeof(frame_number)) return false;
	memcpy(&frame_number, data, sizeof(frame_number));
	return true;
}
void retro_cheat_reset(void) {}
void retro_cheat_set(unsigned index, bool enabled, const char *code) { (void)index; (void)enabled; (void)code; }
bool retro_load_game(const struct retro_game_info *game)
{
	(void)game;
	enum retro_pixel_format fmt = RETRO_PIXEL_FORMAT_XRGB8888;
	return env_cb(RETRO_ENVIRONMENT_SET_PIXEL_FORMAT, &fmt);
}
bool retro_load_game_special(unsigned type, const struct retro_game_info *info, size_t num) { (void)type; (void)info; (void)num; return false; }
void retro_unload_game(void) {}
unsigned retro_get_region(void) { return RETRO_REGION_NTSC; }
void *retro_get_memory_data(unsigned id) { (void)id; return NULL; }
size_t retro_get_memory_size(unsigned id) { (void)id; return 0; }

static inline uint32_t rgb(int r, int g, int b)
{
	return (uint32_t)(r & 255) << 16 | (uint32_t)(g & 255) << 8 | (uint32_t)(b & 255);
}

static uint32_t *frame_buffer_target = frame_buffer;

static void shape(double cx, double cy, double rx, double ry, uint32_t colour)
{
	for (int y = (int)(cy - ry); y <= (int)(cy + ry); y++) {
		for (int x = (int)(cx - rx); x <= (int)(cx + rx); x++) {
			if (x < 0 || y < 0 || x >= W || y >= H) continue;
			const double dx = (x - cx) / rx, dy = (y - cy) / ry;
			if (dx * dx + dy * dy <= 1.0) frame_buffer_target[y * W + x] = colour;
		}
	}
}

/* The scene at any time t (in frames), into `buffer`: retro_run draws it at whole numbers, and
 * a test that wants the frame halfway between two draws it at t + 0.5 (fg_test_render is not part of
 * libretro; a program that loads the core's library can call it). */
void fg_test_render(double t, uint32_t *buffer)
{
	// the scrolling floor: a checkerboard moving 7 px a frame to the left and 2 down
	const double fx = 7.0 * t, fy = 2.0 * t;
	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
			const int u = (int)floor((x + fx) / 40.0), v = (int)floor((y + fy) / 40.0);
			buffer[y * W + x] = ((u + v) & 1) ? rgb(70, 90, 150) : rgb(30, 40, 80);
		}
	}
	uint32_t *keep = frame_buffer_target;
	frame_buffer_target = buffer;
	shape(320 + 250 * sin(t * 0.045), 180 + 90 * cos(t * 0.06), 28, 28, rgb(240, 200, 60));
	shape(100 + fmod(t * 11.0, 440), 260, 36, 18, rgb(220, 70, 70));
	shape(500 - 120 * fabs(sin(t * 0.03)), 80 + 40 * sin(t * 0.2), 22 + 8 * sin(t * 0.1), 22 - 8 * sin(t * 0.1), rgb(80, 220, 120));
	// a fixed HUD: a bar and a block that do not move
	for (int y = 8; y < 24; y++) for (int x = 8; x < 8 + 200; x++) buffer[y * W + x] = rgb(20, 20, 20);
	for (int y = 10; y < 22; y++) for (int x = 10; x < 10 + (int)(196 * (0.5 + 0.5 * sin(t * 0.02))); x++) buffer[y * W + x] = rgb(60, 200, 60);
	for (int y = 8; y < 40; y++) for (int x = W - 70; x < W - 8; x++) buffer[y * W + x] = rgb(250, 250, 250);
	frame_buffer_target = keep;
}

/* FG_TEST_REPEAT=2 makes the scene advance every second frame only: a game at half the core's rate, whose
 * frames reach the frontend twice */
static unsigned repeat_every(void)
{
	const char *s = getenv("FG_TEST_REPEAT");
	const int n = s ? atoi(s) : 1;
	return n >= 1 && n <= 4 ? (unsigned)n : 1;
}

/* FG_TEST_CUT=n switches to another scene (the colours inverted) at core frame n: a cut */
static long cut_frame(void)
{
	const char *s = getenv("FG_TEST_CUT");
	return s ? atol(s) : -1;
}

void retro_run(void)
{
	input_poll_cb();
	const unsigned every = repeat_every();
	fg_test_render((double)(frame_number - frame_number % every) * (60.0 / core_fps()), frame_buffer);
	if (cut_frame() >= 0 && (long)frame_number >= cut_frame()) {
		for (int i = 0; i < W * H; i++) frame_buffer[i] = ~frame_buffer[i] & 0x00ffffff;
	}
	frame_number++;
	video_cb(frame_buffer, W, H, W * sizeof(uint32_t));
	// silence, so the frontend keeps its pace
	static int16_t silence[2 * 800];
	audio_batch_cb(silence, 800);
}
