/*
 * PS5_FrameGen's frame generation for RetroArch's Vulkan driver (gfx/drivers/vulkan.c).
 *
 * With "video_framegen" on and two shader subframes, the driver presents two images for
 * every content frame: first the frame halfway between the previous content frame and this
 * one, made by the trained network (framegen/), then this frame. The functions here keep
 * the previous frame, run the network, and hand the driver an image to show in place of the
 * content frame for the first of the two presents. The first frame, a change of size and
 * anything that breaks the sequence (the menu, a pause) present the content frame alone.
 *
 * Copyright (C) 2026 Mihawk
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef VULKAN_FRAMEGEN_H__
#define VULKAN_FRAMEGEN_H__

#include <stdbool.h>

#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vk_framegen vk_framegen_t;

/* A frame as a sampled image: the content frame, or the generated one handed back. */
struct vk_framegen_frame
{
   VkImage image;
   VkImageView view;
   VkImageLayout layout;
   unsigned width;
   unsigned height;
   VkFormat format;
};

/* Makes the generator for a device with `slots` frames in flight, reading the network from
 * `weights_path`. NULL when the weights cannot be read or are not a network. Images are made
 * on the first frame, once the content's size is known. */
vk_framegen_t *vk_framegen_new(VkDevice device, VkPhysicalDevice gpu, unsigned slots,
      const char *weights_path);
void vk_framegen_free(vk_framegen_t *fg);

/* The history is not valid any more (the frame before is not the one before this one). */
void vk_framegen_invalidate(vk_framegen_t *fg);

/* A new content frame has arrived: records a copy of it (converted to RGBA8) that the frames made for it
 * are made from, so the driver's own texture or the core's image need not outlive the call. Called once
 * for each new frame, outside any render pass, before the generations for it. */
void vk_framegen_hold(vk_framegen_t *fg, VkCommandBuffer cmd, unsigned slot, const struct vk_framegen_frame *cur);

/* Records the generation of the frame at t (0 < t < 1) of the way from the previous new frame to the held
 * one, returning true and, in `out`, a sampled image holding it in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
 * (valid until the command buffer of `slot` is used again). False when no previous frame is held. Called
 * outside any render pass, once for each frame to make. */
bool vk_framegen_generate(vk_framegen_t *fg, VkCommandBuffer cmd, unsigned slot, float t, struct vk_framegen_frame *out);

/* The newest new frame as the held copy, for pixel-art content only (true, and `out` filled in
 * VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL). Real frames are shown from the same image size as the generated ones: a
 * display scale that is not a whole number puts the boundaries of source pixels exactly on pixel centres, and two
 * textures of different sizes round them differently, so the picture's columns would change between real and generated
 * frames. Valid until the command buffer of the slot is used again. */
bool vk_framegen_current(vk_framegen_t *fg, struct vk_framegen_frame *out);

/* The timeline (patches/series, 0121). A new frame starts a group: the pictures between the previous new frame and this
 * one. `gap_s` is how long that interval lasts, in seconds (for a core that returns at its own rate, the calls of the group
 * over the core's frame rate; for a core that blocks for its next frame, what the clock says); `calls` is how many calls of
 * the core the group takes (1 for a core that blocks: a 30 fps game from a core that declares 60 arrives as one call each
 * 2 core frames, and all the group's pictures are presented in that call). The pictures of the group are at the instants the
 * target frame rate puts them at (one every 1/target seconds, `dup` presents each), whatever the game's rate: the game is
 * never run faster or slower to fit a multiple. */
void vk_framegen_begin_group(vk_framegen_t *fg, unsigned calls, double gap_s, double core_fps, double target_fps, unsigned dup);
/* The presents of call `call` of the group (at least one picture's worth, times dup) */
unsigned vk_framegen_call_presents(vk_framegen_t *fg, unsigned call);
/* The picture shown by present `present` (from 1) of a call that has `presents_in_call`: true and its instant in (0, 1) when
 * a frame is to be generated, false when the real frame is to be shown. `final_present` is set at the group's last present:
 * the generator is committed after it. */
bool vk_framegen_picture(vk_framegen_t *fg, unsigned call, unsigned present, unsigned presents_in_call, float *t, bool *final_present);
/* The cost governor: its level (0: as chosen; 1: cheaper; 2: a repeat) and the GPU time of a picture, in ms */
void vk_framegen_governor(const vk_framegen_t *fg, unsigned *level, double *gen_ms);

/* How many new frames have been held */
unsigned long vk_framegen_unique(const vk_framegen_t *fg);

/* The held frame is now the previous one: after the last generation for it. Records nothing. */
void vk_framegen_commit(vk_framegen_t *fg);

/* The method (0 neural, 1 blend, 2 repeat) and the most pixels the generator works at; read every frame. */
void vk_framegen_options(vk_framegen_t *fg, unsigned method, unsigned long cap_pixels);

/* The size the generator works at and how many frames it has made. */
void vk_framegen_stats(const vk_framegen_t *fg, unsigned *width, unsigned *height, unsigned long *generated);

/* The present-stream recorder (see vulkan_framegen.cpp): called for every picture, after the render pass that drew it into
 * the swapchain image `backbuffer` (colour attachment layout, left in it), a copy is taken while a recording is on. */
void vk_framegen_record(vk_framegen_t *fg, VkCommandBuffer cmd, VkImage backbuffer, unsigned width, unsigned height);
/* Asks for a recording of the next `pictures` pictures (0: 240); written to /app0/rec-N, or the folder in PS5_FRAMEGEN_RECORD */
void ps5_fg_record_request(unsigned pictures);

#ifdef __cplusplus
}
#endif

#endif
