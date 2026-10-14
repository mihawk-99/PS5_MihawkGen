/* PS5_FrameGen - a libretro core that wraps another one (FG_REAL_CORE) so that a run on a PC is repeatable: it loads a
 * save state (FG_STATE, at core frame FG_STATE_FRAME) and presses buttons by a script (FG_INPUT, "frame:BUTTON:frames,..."
 * with BUTTON one of B Y SELECT START UP DOWN LEFT RIGHT A X L R). Everything else is forwarded.
 *
 * Copyright (C) 2026 Mihawk
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "libretro.h"

static void *lib;
static retro_input_state_t real_input;
static unsigned long frame;
static int state_done;

#define FN(ret, name, args) static ret(*r_##name) args
FN(void, set_environment, (retro_environment_t));
FN(void, set_video_refresh, (retro_video_refresh_t));
FN(void, set_audio_sample, (retro_audio_sample_t));
FN(void, set_audio_sample_batch, (retro_audio_sample_batch_t));
FN(void, set_input_poll, (retro_input_poll_t));
FN(void, set_input_state, (retro_input_state_t));
FN(void, init, (void));
FN(void, deinit, (void));
FN(unsigned, api_version, (void));
FN(void, get_system_info, (struct retro_system_info *));
FN(void, get_system_av_info, (struct retro_system_av_info *));
FN(void, set_controller_port_device, (unsigned, unsigned));
FN(void, reset, (void));
FN(void, run, (void));
FN(size_t, serialize_size, (void));
FN(bool, serialize, (void *, size_t));
FN(bool, unserialize, (const void *, size_t));
FN(void, cheat_reset, (void));
FN(void, cheat_set, (unsigned, bool, const char *));
FN(bool, load_game, (const struct retro_game_info *));
FN(bool, load_game_special, (unsigned, const struct retro_game_info *, size_t));
FN(void, unload_game, (void));
FN(unsigned, get_region, (void));
FN(void *, get_memory_data, (unsigned));
FN(size_t, get_memory_size, (unsigned));

static void load_real(void)
{
    if (lib)
        return;
    const char *path = getenv("FG_REAL_CORE");
    lib = path ? dlopen(path, RTLD_NOW | RTLD_LOCAL) : NULL;
    if (!lib) {
        fprintf(stderr, "fg_proxy: cannot load FG_REAL_CORE (%s)\n", path ? dlerror() : "unset");
        exit(2);
    }
#define L(n) r_##n = dlsym(lib, "retro_" #n)
    L(set_environment); L(set_video_refresh); L(set_audio_sample); L(set_audio_sample_batch); L(set_input_poll);
    L(set_input_state); L(init); L(deinit); L(api_version); L(get_system_info); L(get_system_av_info);
    L(set_controller_port_device); L(reset); L(run); L(serialize_size); L(serialize); L(unserialize); L(cheat_reset);
    L(cheat_set); L(load_game); L(load_game_special); L(unload_game); L(get_region); L(get_memory_data); L(get_memory_size);
}

static int button_id(const char *name)
{
    static const char *const names[] = { "B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R" };
    for (int i = 0; i < 12; i++)
        if (!strcmp(name, names[i]))
            return i;
    return -1;
}

/* Capture (FG_CAPTURE=file): the frames the core makes, as RGB bytes behind a 12-byte header (width, height, count),
 * for training the network on real clips. FG_CAPTURE_SKIP frames are let go first, FG_CAPTURE_COUNT are kept. The
 * files stay on this PC, outside every repository. */
static retro_environment_t real_env;
static retro_video_refresh_t real_video;
static unsigned pixel_format;
static FILE *capture;
static unsigned long captured, capture_seen;
static unsigned cap_w, cap_h;

static bool env_cb(unsigned cmd, void *data)
{
    if (cmd == RETRO_ENVIRONMENT_SET_PIXEL_FORMAT && data)
        pixel_format = *(unsigned *)data;
    return real_env(cmd, data);
}

static void video_cb(const void *data, unsigned w, unsigned h, size_t pitch)
{
    const char *path = getenv("FG_CAPTURE");
    if (path && data && data != (void *)-1) {
        const unsigned long skip = getenv("FG_CAPTURE_SKIP") ? strtoul(getenv("FG_CAPTURE_SKIP"), NULL, 10) : 0;
        const unsigned long count = getenv("FG_CAPTURE_COUNT") ? strtoul(getenv("FG_CAPTURE_COUNT"), NULL, 10) : 64;
        if (capture_seen++ >= skip && captured < count) {
            if (!capture) {
                capture = fopen(path, "wb");
                cap_w = w; cap_h = h;
                unsigned header[3] = { w, h, 0 };
                fwrite(header, 4, 3, capture);
            }
            if (capture && w == cap_w && h == cap_h) {
                unsigned char *row = malloc((size_t)w * 3);
                for (unsigned y = 0; y < h; y++) {
                    const unsigned char *src = (const unsigned char *)data + y * pitch;
                    for (unsigned x = 0; x < w; x++) {
                        unsigned r, g, b;
                        if (pixel_format == RETRO_PIXEL_FORMAT_XRGB8888) {
                            const uint32_t v = ((const uint32_t *)src)[x];
                            r = (v >> 16) & 255; g = (v >> 8) & 255; b = v & 255;
                        } else if (pixel_format == RETRO_PIXEL_FORMAT_RGB565) {
                            const uint16_t v = ((const uint16_t *)src)[x];
                            r = ((v >> 11) & 31) * 255 / 31; g = ((v >> 5) & 63) * 255 / 63; b = (v & 31) * 255 / 31;
                        } else {
                            const uint16_t v = ((const uint16_t *)src)[x];
                            r = ((v >> 10) & 31) * 255 / 31; g = ((v >> 5) & 31) * 255 / 31; b = (v & 31) * 255 / 31;
                        }
                        row[x * 3] = (unsigned char)r; row[x * 3 + 1] = (unsigned char)g; row[x * 3 + 2] = (unsigned char)b;
                    }
                    fwrite(row, 1, (size_t)w * 3, capture);
                }
                free(row);
                captured++;
                if (captured == count) {
                    unsigned header[3] = { w, h, (unsigned)captured };
                    fseek(capture, 0, SEEK_SET);
                    fwrite(header, 4, 3, capture);
                    fclose(capture);
                    capture = NULL;
                    _exit(0);              /* (a PC capture run: nothing more to do) */
                }
            }
        }
    }
    real_video(data, w, h, pitch);
}

/* Random play (FG_RANDOM=seed): a direction held for a while, now and then a button. */
static void capture_finish(void)
{
    if (capture) {
        unsigned header[3] = { cap_w, cap_h, (unsigned)captured };
        fseek(capture, 0, SEEK_SET);
        fwrite(header, 4, 3, capture);
        fclose(capture);
        capture = NULL;
    }
}

static int16_t stick[4];                          /* left x, y and right x, y of random play */
static int16_t random_mask(void)
{
    static unsigned long seed, until;
    static int16_t mask;
    if (!getenv("FG_RANDOM"))
        return 0;
    if (!seed)
        seed = strtoul(getenv("FG_RANDOM"), NULL, 10) * 2654435761ul + 12345;
    if (frame >= until) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned r = (unsigned)(seed >> 33);
        mask = 0;
        const unsigned dir = r % 6;                   /* up down left right, or none twice */
        if (dir < 4)
            mask |= (int16_t)(1 << (4 + dir));
        if ((r >> 8) % 4 == 0)
            mask |= (int16_t)(1 << ((r >> 12) % 2 ? 8 : 0));   /* A or B */
        if ((r >> 16) % 24 == 0)
            mask |= (int16_t)(1 << 3);                /* START, rarely */
        until = frame + 10 + (r >> 20) % 40;
        {   /* the sticks too: a direction at three quarters most of the time, the camera now and then */
            const unsigned q = (unsigned)(seed >> 40);
            const double angle = (double)(q % 360) * 3.14159265 / 180.0;
            const int still = (q >> 9) % 5 == 0;
            stick[0] = still ? 0 : (int16_t)(24000 * __builtin_cos(angle));
            stick[1] = still ? 0 : (int16_t)(24000 * __builtin_sin(angle));
            stick[2] = ((q >> 12) % 3 == 0) ? (int16_t)(16000 * __builtin_cos(angle + 1.0)) : 0;
            stick[3] = ((q >> 12) % 3 == 0) ? (int16_t)(16000 * __builtin_sin(angle + 1.0)) : 0;
        }
    }
    return mask;
}

static int16_t input_state(unsigned port, unsigned device, unsigned index, unsigned id)
{
    int16_t value = real_input ? real_input(port, device, index, id) : 0;
    const char *script = getenv("FG_INPUT");
    if (port == 0 && device == RETRO_DEVICE_ANALOG && getenv("FG_RANDOM")) {
        random_mask();                        /* (moves the sticks on) */
        return stick[(index == RETRO_DEVICE_INDEX_ANALOG_RIGHT ? 2 : 0) + (id == RETRO_DEVICE_ID_ANALOG_Y ? 1 : 0)];
    }
    int16_t random_part = 0;
    if (port == 0 && device == RETRO_DEVICE_JOYPAD && getenv("FG_RANDOM"))
        random_part = random_mask();       /* (random play and the script's presses together) */
    if (port != 0 || device != RETRO_DEVICE_JOYPAD || !script) {
        if (random_part)
            return id == RETRO_DEVICE_ID_JOYPAD_MASK ? (value | random_part) : ((random_part >> id) & 1 ? 1 : value);
        return value;
    }
    char copy[512];
    strncpy(copy, script, sizeof(copy) - 1);
    copy[sizeof(copy) - 1] = 0;
    int16_t mask = random_part;
    for (char *item = strtok(copy, ","); item; item = strtok(NULL, ",")) {
        unsigned long start, length;
        char name[16];
        if (sscanf(item, "%lu:%15[A-Z]:%lu", &start, name, &length) == 3 && frame >= start && frame < start + length) {
            const int b = button_id(name);
            if (b >= 0)
                mask |= (int16_t)(1 << b);
        }
    }
    if (id == RETRO_DEVICE_ID_JOYPAD_MASK)       /* cores that ask for all the buttons at once */
        return value | mask;
    return (mask >> id) & 1 ? 1 : value;
}

void retro_set_environment(retro_environment_t cb) { load_real(); real_env = cb; r_set_environment(env_cb); }
void retro_set_video_refresh(retro_video_refresh_t cb) { load_real(); real_video = cb; r_set_video_refresh(video_cb); }
void retro_set_audio_sample(retro_audio_sample_t cb) { load_real(); r_set_audio_sample(cb); }
void retro_set_audio_sample_batch(retro_audio_sample_batch_t cb) { load_real(); r_set_audio_sample_batch(cb); }
void retro_set_input_poll(retro_input_poll_t cb) { load_real(); r_set_input_poll(cb); }
void retro_set_input_state(retro_input_state_t cb) { load_real(); real_input = cb; r_set_input_state(input_state); }
void retro_init(void) { load_real(); r_init(); }
void retro_deinit(void) { capture_finish(); r_deinit(); }
unsigned retro_api_version(void) { load_real(); return r_api_version(); }
void retro_get_system_info(struct retro_system_info *info) { load_real(); r_get_system_info(info); }
void retro_get_system_av_info(struct retro_system_av_info *info)
{
    r_get_system_av_info(info);
    if (getenv("FG_FPS"))                    /* a game at another rate, for runs on a PC (the core still runs as fast as it can) */
        info->timing.fps = atof(getenv("FG_FPS"));
}
void retro_set_controller_port_device(unsigned p, unsigned d) { r_set_controller_port_device(p, d); }
void retro_reset(void) { r_reset(); }
void retro_run(void)
{
    const char *state = getenv("FG_STATE");
    if (state && !state_done && frame >= (getenv("FG_STATE_FRAME") ? strtoul(getenv("FG_STATE_FRAME"), NULL, 10) : 5)) {
        state_done = 1;
        FILE *f = fopen(state, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long n = ftell(f);
            fseek(f, 0, SEEK_SET);
            void *data = malloc((size_t)n);
            if (fread(data, 1, (size_t)n, f) == (size_t)n)
                fprintf(stderr, "fg_proxy: state %s: %s\n", state, r_unserialize(data, (size_t)n) ? "loaded" : "REFUSED");
            free(data);
            fclose(f);
        }
    }
    r_run();
    frame++;
}
size_t retro_serialize_size(void) { return r_serialize_size(); }
bool retro_serialize(void *d, size_t n) { return r_serialize(d, n); }
bool retro_unserialize(const void *d, size_t n) { return r_unserialize(d, n); }
void retro_cheat_reset(void) { r_cheat_reset(); }
void retro_cheat_set(unsigned i, bool e, const char *c) { r_cheat_set(i, e, c); }
bool retro_load_game(const struct retro_game_info *g) { return r_load_game(g); }
bool retro_load_game_special(unsigned t, const struct retro_game_info *g, size_t n) { return r_load_game_special(t, g, n); }
void retro_unload_game(void) { r_unload_game(); }
unsigned retro_get_region(void) { return r_get_region(); }
void *retro_get_memory_data(unsigned id) { return r_get_memory_data(id); }
size_t retro_get_memory_size(unsigned id) { return r_get_memory_size(id); }
