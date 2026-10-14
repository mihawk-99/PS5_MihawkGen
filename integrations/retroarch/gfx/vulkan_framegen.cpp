/*
 * PS5_FrameGen's frame generation for RetroArch's Vulkan driver: see vulkan_framegen.h.
 *
 * Copyright (C) 2026 Mihawk
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "vulkan_framegen.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <time.h>

#include "framegen.h"

extern "C" {
#include "verbosity.h"
}

namespace {

constexpr unsigned kMaxSlots = 8;

struct Target {
	VkImage image{ VK_NULL_HANDLE };
	VkDeviceMemory memory{ VK_NULL_HANDLE };
	VkImageView view{ VK_NULL_HANDLE };
};

} // namespace

// A debugging aid: with PS5_FRAMEGEN_DUMP=<folder> in the environment, the frames made for each of the first 60
// content frames (gen_<content>_<n>.rgba and their time, .t) and the content frames (cur_<content>.rgba) are
// written there, raw RGBA8, the size in size.txt, so a test can score them against frames known exactly.
struct Dump {
	std::string dir;
	VkBuffer buffer[2]{};
	VkDeviceMemory memory[2]{};
	void *mapped[2]{};
	unsigned width{ 0 }, height{ 0 };
	int pending_gen{ -1 };       // the content frame whose generated copy waits to be written, and its number within it
	int pending_gen_index{ 0 };
	float pending_t{ 0.0f };
	int pending_cur{ -1 };       // the content frame whose copy waits to be written
	unsigned long gen_present{ 0 }, cur_present{ 0 };   // the present (vk_framegen::presents) each copy was recorded in
	VkCommandBuffer cmd_gen{ VK_NULL_HANDLE };   // the command buffers that record those copies: a copy is read
	VkCommandBuffer cmd_cur{ VK_NULL_HANDLE };   // back only once a later command buffer is being recorded
	int count{ 0 };              // content frames committed
	int gen_in_frame{ 0 };       // generations so far in the current content frame
	int skip{ 0 };               // PS5_FRAMEGEN_DUMP_SKIP: content frames to let pass before the dump starts
	int skipped{ 0 };
	int limit{ 60 };             // PS5_FRAMEGEN_DUMP_COUNT: how many content frames to write
	int stride{ 0 };             // PS5_FRAMEGEN_DUMP_STRIDE: new frames between one window of the dump and the next (0: one window)
	int window{ 0 };
	int since_end{ 0 };
	std::string base;            // the folder the windows' folders (w00, w01, ...) are in
};

// The present-stream recorder: the pictures the display is given, one after another (the real frames and the generated ones
// as they were shown), taken from the swapchain image just before it is presented, scaled down to a small size, into one
// host-visible buffer and written out when the window is full: <folder>/rec-N/frames.rgba (raw RGBA8, one picture after the
// other), size.txt and index.txt (a line a picture: its place, the time it was recorded in ns, 1 if generated, its instant
// t, the new frames held so far, the governor's level). Asked for by the pad script's RECORD (the PS5 title) or by
// PS5_FRAMEGEN_RECORD=<folder> with PS5_FRAMEGEN_RECORD_AT=<picture to begin at> and PS5_FRAMEGEN_RECORD_COUNT (a PC run).
struct Recorder {
	bool active{ false };
	std::string dir;
	unsigned width{ 0 }, height{ 0 };   // the recorded size
	unsigned capacity{ 0 }, count{ 0 };
	VkImage image{ VK_NULL_HANDLE };
	VkDeviceMemory image_memory{ VK_NULL_HANDLE };
	VkBuffer buffer{ VK_NULL_HANDLE };
	VkDeviceMemory buffer_memory{ VK_NULL_HANDLE };
	unsigned char *mapped{ nullptr };
	struct Meta {
		unsigned long place;
		unsigned long long ns;
		bool generated;
		float t;
		unsigned long unique;
		unsigned level;
	};
	std::vector<Meta> meta;
	unsigned long serial{ 0 };          // pictures seen since the start
	unsigned long runs{ 0 };            // recordings made
	unsigned long env_at{ 0 };          // PS5_FRAMEGEN_RECORD: the picture to begin at (0: not asked for)
	unsigned env_count{ 240 };
	bool env_done{ false };
};

struct vk_framegen {
	VkDevice device{ VK_NULL_HANDLE };
	VkPhysicalDevice gpu{ VK_NULL_HANDLE };
	unsigned slots{ 0 };
	std::vector<char> weights;
	fg_context *ctx{ nullptr };
	unsigned width{ 0 }, height{ 0 };          // the size the generator works at
	unsigned src_width{ 0 }, src_height{ 0 };  // the size of the frames it is given

	Target frames[2];               // the previous new frame and the held one, RGBA8
	unsigned held{ 0 };             // which of them is the held one
	bool has_held{ false };
	unsigned latest{ 0 };           // the image holding the newest new frame (it stays valid after the commit)
	bool latest_valid{ false };
	Target generated[kMaxSlots];    // the frame made for each slot
	bool history_valid{ false };
	bool pixel_art{ false };        // the frames are low resolution: generated at twice their size, in whole pixels
	unsigned method{ 0 };           // 0 neural, 1 blend, 2 repeat
	unsigned long cap{ 3840ul * 2160ul };  // the most pixels the generator works at
	unsigned long cap_made{ 0 };
	unsigned long generated_count{ 0 };
	unsigned long presents{ 0 };    // presents that asked for a frame (counted when the generation call ends)
	bool generated_ready[kMaxSlots]{};
	// The timeline (patches/series, 0121): the pictures shown between the previous new frame and this one, at the
	// instants the target frame rate puts them at, whatever the game's own rate.
	// The cost governor: the GPU time of what the shim records is measured with timestamps and the method steps down to a
	// cheaper one when it takes more of a picture's time than it may
	VkQueryPool qpool{ VK_NULL_HANDLE };
	double ts_ns{ 1.0 };                // a timestamp tick, in nanoseconds
	bool q_pending[kMaxSlots][2]{};     // a generation / a hold of the slot has timestamps to read
	double gen_ms{ 0.0 }, hold_ms{ 0.0 };   // averages of what has been read
	double period_ms{ 1000.0 / 119.88 };    // a picture's time: 1 / the target rate
	unsigned level{ 0 };                // 0: as chosen; 1: cheaper (pixel art: motion pass alone; else blend); 2: repeat
	unsigned over{ 0 }, under{ 0 };     // groups in a row above the limit / far below it
	unsigned up_after{ 600 };           // groups far below the limit it takes to step up: doubles when stepping up only brought it back down
	unsigned since_up{ 1000000 };       // groups since the last step up
	bool tiles_found{ false };          // the motion search of the held pair has been recorded (it does not depend on t)
	unsigned long unique_count{ 0 };    // new frames held
	double phase{ 1.0 };                // when the next picture falls, in new-frame intervals after the last new frame
	unsigned group_period{ 1 };         // calls in the group
	double group_gap_s{ 0.0 };          // seconds the group's interval lasts
	unsigned nominal{ 1 };              // pictures a call shows when the group has no more to say (a gap longer than was predicted)
	unsigned dup{ 1 };                  // presents of each picture
	std::vector<float> times;           // the instants of the group's pictures: (0, 1], the last real frame at 1

	Dump dump;
	Recorder rec;
	// The hold trace (PS5_FRAMEGEN_TRACE=1, or /app0/fg-trace.txt): for each new frame, the image the core handed over and a hash of a strip
	// of its pixels, read three holds later, so that the order the core delivers its frames in can be read from the log
	struct Trace {
		int enabled{ -1 };
		VkBuffer buffer[4]{};
		VkDeviceMemory memory[4]{};
		void *mapped[4]{};
		unsigned long number[4]{};
		bool made{ false };
	} trace;
	bool pic_generated{ false };   // the picture being recorded is a generated one, and its instant
	float pic_t{ 1.0f };
};

static uint32_t find_memory(const VkPhysicalDeviceMemoryProperties &mem, uint32_t bits, VkMemoryPropertyFlags want)
{
	for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
		if ((bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & want) == want) {
			return i;
		}
	}
	return UINT32_MAX;
}

static void destroy_target(VkDevice device, Target &t)
{
	if (t.view) vkDestroyImageView(device, t.view, nullptr);
	if (t.image) vkDestroyImage(device, t.image, nullptr);
	if (t.memory) vkFreeMemory(device, t.memory, nullptr);
	t = {};
}

static void destroy_resources(vk_framegen *fg)
{
	if (fg->device) {
		vkDeviceWaitIdle(fg->device);
		if (fg->qpool) {
			vkDestroyQueryPool(fg->device, fg->qpool, nullptr);
			fg->qpool = VK_NULL_HANDLE;
		}
		memset(fg->q_pending, 0, sizeof(fg->q_pending));
	}
	fg_context_destroy(fg->ctx);
	fg->ctx = nullptr;
	destroy_target(fg->device, fg->frames[0]);
	destroy_target(fg->device, fg->frames[1]);
	fg->has_held = false;
	for (Target &t : fg->generated) {
		destroy_target(fg->device, t);
	}
	fg->history_valid = false;
	memset(fg->generated_ready, 0, sizeof(fg->generated_ready));
}

static bool make_target(vk_framegen *fg, Target &t, unsigned w, unsigned h)
{
	const VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM,
		{ w, h, 1 }, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, VK_IMAGE_LAYOUT_UNDEFINED };
	if (vkCreateImage(fg->device, &imageInfo, nullptr, &t.image) != VK_SUCCESS) return false;
	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(fg->device, t.image, &req);
	VkPhysicalDeviceMemoryProperties mem;
	vkGetPhysicalDeviceMemoryProperties(fg->gpu, &mem);
	const uint32_t type = find_memory(mem, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (type == UINT32_MAX) return false;
	const VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, type };
	if (vkAllocateMemory(fg->device, &ai, nullptr, &t.memory) != VK_SUCCESS) return false;
	if (vkBindImageMemory(fg->device, t.image, t.memory, 0) != VK_SUCCESS) return false;
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = t.image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
	viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	return vkCreateImageView(fg->device, &viewInfo, nullptr, &t.view) == VK_SUCCESS;
}

static bool make_resources(vk_framegen *fg, unsigned w, unsigned h, unsigned source_w, unsigned source_h)
{
	destroy_resources(fg);
	const fg_create_info info{ fg->device, fg->gpu, w, h, fg->slots, VK_FORMAT_R8G8B8A8_UNORM, source_w, source_h };
	char error[256] = "";
	if (fg_context_create(&info, &fg->ctx, error, sizeof(error)) != VK_SUCCESS) {
		RARCH_ERR("[FrameGen] %s\n", error);
		return false;
	}
	if (fg_context_load_network(fg->ctx, fg->weights.data(), fg->weights.size(), error, sizeof(error)) != VK_SUCCESS) {
		RARCH_ERR("[FrameGen] %s\n", error);
		return false;
	}
	if (!make_target(fg, fg->frames[0], w, h) || !make_target(fg, fg->frames[1], w, h)) return false;
	for (unsigned s = 0; s < fg->slots; s++) {
		if (!make_target(fg, fg->generated[s], w, h)) return false;
	}
	fg->width = w;
	fg->height = h;
	RARCH_LOG("[FrameGen] network ready for %ux%u frames.\n", w, h);
	return true;
}

static void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst,
	VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
{
	VkImageMemoryBarrier b{};
	b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	b.srcAccessMask = src;
	b.dstAccessMask = dst;
	b.oldLayout = from;
	b.newLayout = to;
	b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image = image;
	b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

static void dump_destroy(vk_framegen *fg)
{
	Dump &d = fg->dump;
	for (int i = 0; i < 2; i++) {
		if (d.buffer[i]) vkDestroyBuffer(fg->device, d.buffer[i], nullptr);
		if (d.memory[i]) vkFreeMemory(fg->device, d.memory[i], nullptr);
		d.buffer[i] = VK_NULL_HANDLE;
		d.memory[i] = VK_NULL_HANDLE;
		d.mapped[i] = nullptr;
	}
}

static bool dump_prepare(vk_framegen *fg, unsigned w, unsigned h)
{
	Dump &d = fg->dump;
	dump_destroy(fg);
	d.width = w;
	d.height = h;
	VkPhysicalDeviceMemoryProperties mem;
	vkGetPhysicalDeviceMemoryProperties(fg->gpu, &mem);
	for (int i = 0; i < 2; i++) {
		VkBufferCreateInfo bi{};
		bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bi.size = VkDeviceSize(w) * h * 4;
		bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		if (vkCreateBuffer(fg->device, &bi, nullptr, &d.buffer[i]) != VK_SUCCESS) return false;
		VkMemoryRequirements req;
		vkGetBufferMemoryRequirements(fg->device, d.buffer[i], &req);
		const uint32_t type = find_memory(mem, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		if (type == UINT32_MAX) return false;
		const VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, type };
		if (vkAllocateMemory(fg->device, &ai, nullptr, &d.memory[i]) != VK_SUCCESS) return false;
		if (vkBindBufferMemory(fg->device, d.buffer[i], d.memory[i], 0) != VK_SUCCESS) return false;
		if (vkMapMemory(fg->device, d.memory[i], 0, VK_WHOLE_SIZE, 0, &d.mapped[i]) != VK_SUCCESS) return false;
	}
	char path[512];
	snprintf(path, sizeof(path), "%s/size.txt", d.dir.c_str());
	if (FILE *f = fopen(path, "w")) {
		fprintf(f, "%u %u\n", w, h);
		fclose(f);
	}
	return true;
}

// Writes what the copies recorded since the last flush hold (the device is idle for it)
static void dump_flush(vk_framegen *fg, VkCommandBuffer recording)
{
	Dump &d = fg->dump;
	// A copy is ready once a later present is being recorded (its own command buffer was submitted by then, whether or
	// not the driver hands the same command buffer out again)
	(void)recording;
	const bool gen_ready = d.pending_gen >= 0 && d.gen_present < fg->presents;
	const bool cur_ready = d.pending_cur >= 0 && d.cur_present < fg->presents;
	if (!gen_ready && !cur_ready) return;
	vkDeviceWaitIdle(fg->device);
	const size_t bytes = size_t(d.width) * d.height * 4;
	char path[512];
	if (gen_ready) {
		snprintf(path, sizeof(path), "%s/gen_%04d_%d.rgba", d.dir.c_str(), d.pending_gen, d.pending_gen_index);
		if (FILE *f = fopen(path, "wb")) {
			fwrite(d.mapped[0], 1, bytes, f);
			fclose(f);
		}
		snprintf(path, sizeof(path), "%s/gen_%04d_%d.t", d.dir.c_str(), d.pending_gen, d.pending_gen_index);
		if (FILE *f = fopen(path, "w")) {
			fprintf(f, "%.6f\n", d.pending_t);
			fclose(f);
		}
		d.pending_gen = -1;
	}
	if (cur_ready) {
		snprintf(path, sizeof(path), "%s/cur_%04d.rgba", d.dir.c_str(), d.pending_cur);
		if (FILE *f = fopen(path, "wb")) {
			fwrite(d.mapped[1], 1, bytes, f);
			fclose(f);
		}
		d.pending_cur = -1;
	}
}

static void dump_buffers(vk_framegen *fg)
{
	Dump &d = fg->dump;
	if (!d.buffer[0]) dump_prepare(fg, fg->width, fg->height);
}

// Copies the frame just generated (the one in the dump buffer 0) for writing at the next flush
static void dump_record(vk_framegen *fg, VkCommandBuffer cmd, bool generated, const Target &gen, float t)
{
	Dump &d = fg->dump;
	static const bool cur_only = getenv("PS5_FRAMEGEN_DUMP_CURONLY") != nullptr;   // (clips for training: the real frames alone)
	if (cur_only || !generated || d.count >= d.limit || d.count == 0) return;
	dump_buffers(fg);
	if (!d.buffer[0]) return;
	dump_flush(fg, cmd);
	const VkBufferImageCopy region{ 0, 0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { fg->width, fg->height, 1 } };
	image_barrier(cmd, gen.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
		VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	vkCmdCopyImageToBuffer(cmd, gen.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, d.buffer[0], 1, &region);
	image_barrier(cmd, gen.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
		VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	// content frame d.count; its generations are numbered from 0 within it
	d.cmd_gen = cmd;
	d.gen_present = fg->presents;
	d.pending_gen = d.count - 1;
	d.pending_gen_index = d.gen_in_frame++;
	d.pending_t = t;
}

// Copies the new frame (the held image) for writing at the next flush; its number is the group's
static void dump_hold(vk_framegen *fg, VkCommandBuffer cmd)
{
	Dump &d = fg->dump;
	if (d.count >= d.limit) {
		if (d.stride > 0 && d.buffer[1]) {
			dump_flush(fg, cmd);      // (the window's last copy)
			if (++d.since_end >= d.stride) {
				// the next window, in a folder of its own
				char name[32];
				snprintf(name, sizeof(name), "/w%02d", ++d.window);
				d.dir = d.base + name;
				mkdir(d.dir.c_str(), 0755);
				d.count = 0;
				d.since_end = 0;
				char path[512];
				snprintf(path, sizeof(path), "%s/size.txt", d.dir.c_str());
				if (FILE *f = fopen(path, "w")) {
					fprintf(f, "%u %u\n", d.width, d.height);
					fclose(f);
				}
			}
		}
		return;
	}
	if (d.skipped < d.skip) {
		d.skipped++;
		return;
	}
	dump_buffers(fg);
	if (!d.buffer[1]) return;
	dump_flush(fg, cmd);
	const Target &held = fg->frames[fg->held];
	const VkBufferImageCopy region{ 0, 0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { fg->width, fg->height, 1 } };
	image_barrier(cmd, held.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
		VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	vkCmdCopyImageToBuffer(cmd, held.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, d.buffer[1], 1, &region);
	image_barrier(cmd, held.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
		VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	d.cmd_cur = cmd;
	d.cur_present = fg->presents;
	d.pending_cur = d.count++;
	d.gen_in_frame = 0;
}

vk_framegen_t *vk_framegen_new(VkDevice device, VkPhysicalDevice gpu, unsigned slots, const char *weights_path)
{
	if (!device || !gpu || slots == 0 || slots > kMaxSlots || !weights_path) {
		return nullptr;
	}
	FILE *file = fopen(weights_path, "rb");
	if (!file) {
		RARCH_ERR("[FrameGen] Cannot read the network \"%s\".\n", weights_path);
		return nullptr;
	}
	vk_framegen *fg = new vk_framegen;
	fseek(file, 0, SEEK_END);
	const long size = ftell(file);
	fseek(file, 0, SEEK_SET);
	fg->weights.resize(size > 0 ? size_t(size) : 0);
	if (size <= 0 || fread(fg->weights.data(), 1, fg->weights.size(), file) != fg->weights.size()) {
		fclose(file);
		delete fg;
		return nullptr;
	}
	fclose(file);
	fg->device = device;
	fg->gpu = gpu;
	fg->slots = slots;
	if (const char *rec = getenv("PS5_FRAMEGEN_RECORD")) {
		fg->rec.dir = rec;
		fg->rec.env_at = getenv("PS5_FRAMEGEN_RECORD_AT") ? strtoul(getenv("PS5_FRAMEGEN_RECORD_AT"), nullptr, 10) : 600ul;
		if (getenv("PS5_FRAMEGEN_RECORD_COUNT")) fg->rec.env_count = unsigned(atoi(getenv("PS5_FRAMEGEN_RECORD_COUNT")));
	}
	if (const char *dump = getenv("PS5_FRAMEGEN_DUMP")) {
		fg->dump.dir = dump;
		if (const char *skip = getenv("PS5_FRAMEGEN_DUMP_SKIP")) fg->dump.skip = atoi(skip);
		if (const char *count = getenv("PS5_FRAMEGEN_DUMP_COUNT")) fg->dump.limit = atoi(count);
		if (const char *stride = getenv("PS5_FRAMEGEN_DUMP_STRIDE")) {
			fg->dump.stride = atoi(stride);
			if (fg->dump.stride > 0) {
				fg->dump.base = fg->dump.dir;
				fg->dump.dir = fg->dump.base + "/w00";
				mkdir(fg->dump.dir.c_str(), 0755);
			}
		}
	}
	return fg;
}

static void record_free(vk_framegen *fg);

void vk_framegen_free(vk_framegen_t *fg)
{
	if (!fg) return;
	destroy_resources(fg);
	dump_destroy(fg);
	record_free(fg);
	delete fg;
}

void vk_framegen_options(vk_framegen_t *fg, unsigned method, unsigned long cap_pixels)
{
	if (!fg) return;
	fg->method = method % 3;
	if (cap_pixels >= 1280ul * 720ul) fg->cap = cap_pixels;
}

void vk_framegen_governor(const vk_framegen_t *fg, unsigned *level, double *gen_ms)
{
	if (!fg) return;
	*level = fg->level;
	*gen_ms = fg->gen_ms + fg->hold_ms / std::max(1.0, double(fg->times.size()));
}

void vk_framegen_stats(const vk_framegen_t *fg, unsigned *width, unsigned *height, unsigned long *generated)
{
	if (!fg) return;
	*width = fg->width;
	*height = fg->height;
	*generated = fg->generated_count;
}

void vk_framegen_invalidate(vk_framegen_t *fg)
{
	if (fg) {
		fg->history_valid = false;
		fg->has_held = false;
	}
}

static bool ensure_resources(vk_framegen *fg, const vk_framegen_frame *cur)
{
	if (!cur || !cur->view || cur->width == 0 || cur->height == 0) {
		return false;
	}
	if (fg->src_width != cur->width || fg->src_height != cur->height || fg->cap_made != fg->cap || !fg->ctx) {
		// A core may render at many times the screen's size (16x internal resolution): the generator works at
		// no more than a 4K frame's pixels, which the held copy averages down to.
		// Pixel art (frames no taller than the network's working height) is generated at 3 or 4 times its size,
		// each source pixel a block of pixels, so a motion of a fraction of a pixel between two frames (a camera
		// that eases, a diagonal step) lands on a block edge and stays sharp, and the half step is exact to a
		// quarter of a pixel (the final pass warps such frames in whole pixels).
		const unsigned scale = cur->height <= 288 ? std::min(4u, std::max(2u, 1080u / cur->height)) : 1u;
		unsigned w = cur->width * scale, h = cur->height * scale;
		fg->pixel_art = scale > 1;
		const double cap = double(fg->cap), area = double(w) * double(h);
		if (area > cap) {
			const double f = std::sqrt(cap / area);
			w = unsigned(w * f) & ~1u;
			h = unsigned(h * f) & ~1u;
		}
		fg->src_width = cur->width;
		fg->src_height = cur->height;
		fg->cap_made = fg->cap;
		// pixel art that is not averaged down (the frames are a whole-number enlargement): the motion pass's source size
		const bool exact = scale > 1 && w == cur->width * scale && h == cur->height * scale;
		if (!make_resources(fg, w, h, exact ? cur->width : 0, exact ? cur->height : 0)) {
			destroy_resources(fg);
			fg->width = fg->height = 0;
			fg->src_width = fg->src_height = 0;
			return false;
		}
	}
	return true;
}

// Timestamps around a recording: two queries for each slot and kind (0 a generation, 1 a hold)
static void query_begin(vk_framegen *fg, VkCommandBuffer cmd, unsigned slot, unsigned kind)
{
	if (!fg->qpool) {
		VkPhysicalDeviceProperties props{};
		vkGetPhysicalDeviceProperties(fg->gpu, &props);
		VkQueryPoolCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		info.queryType = VK_QUERY_TYPE_TIMESTAMP;
		info.queryCount = kMaxSlots * 4;
		fg->ts_ns = double(props.limits.timestampPeriod);
		if (vkCreateQueryPool(fg->device, &info, nullptr, &fg->qpool) != VK_SUCCESS) {
			fg->qpool = VK_NULL_HANDLE;
			return;
		}
	}
	const unsigned base = slot * 4 + kind * 2;
	if (fg->q_pending[slot][kind]) {
		uint64_t r[4] = {};
		if (vkGetQueryPoolResults(fg->device, fg->qpool, base, 2, sizeof(r), r, 16, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS
			&& r[1] && r[3] && r[2] >= r[0]) {
			const double ms = double(r[2] - r[0]) * fg->ts_ns / 1e6;
			double &avg = kind == 0 ? fg->gen_ms : fg->hold_ms;
			avg = avg == 0.0 ? ms : avg * 0.9 + ms * 0.1;
			fg->q_pending[slot][kind] = false;
		}
	}
	vkCmdResetQueryPool(cmd, fg->qpool, base, 2);
	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, fg->qpool, base);
}

static void query_end(vk_framegen *fg, VkCommandBuffer cmd, unsigned slot, unsigned kind)
{
	if (!fg->qpool) return;
	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, fg->qpool, slot * 4 + kind * 2 + 1);
	fg->q_pending[slot][kind] = true;
}


static bool trace_enabled(vk_framegen *fg)
{
	if (fg->trace.enabled < 0) {
		FILE *f = fopen("/app0/fg-trace.txt", "r");
		fg->trace.enabled = (getenv("PS5_FRAMEGEN_TRACE") || f) ? 1 : 0;
		if (f) fclose(f);
	}
	return fg->trace.enabled == 1;
}

static void trace_hold(vk_framegen *fg, VkCommandBuffer cmd, const Target &held, const vk_framegen_frame *cur)
{
	auto &t = fg->trace;
	const unsigned strip_w = 128, strip_h = 4;
	if (!t.made) {
		VkPhysicalDeviceMemoryProperties mem;
		vkGetPhysicalDeviceMemoryProperties(fg->gpu, &mem);
		for (int i = 0; i < 4; i++) {
			VkBufferCreateInfo bi{};
			bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bi.size = strip_w * strip_h * 4;
			bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			if (vkCreateBuffer(fg->device, &bi, nullptr, &t.buffer[i]) != VK_SUCCESS) { t.enabled = 0; return; }
			VkMemoryRequirements req;
			vkGetBufferMemoryRequirements(fg->device, t.buffer[i], &req);
			const uint32_t type = find_memory(mem, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
			const VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, type };
			if (type == UINT32_MAX || vkAllocateMemory(fg->device, &ai, nullptr, &t.memory[i]) != VK_SUCCESS
				|| vkBindBufferMemory(fg->device, t.buffer[i], t.memory[i], 0) != VK_SUCCESS
				|| vkMapMemory(fg->device, t.memory[i], 0, VK_WHOLE_SIZE, 0, &t.mapped[i]) != VK_SUCCESS) { t.enabled = 0; return; }
			memset(t.mapped[i], 0, strip_w * strip_h * 4);
		}
		t.made = true;
	}
	const unsigned n = unsigned(fg->unique_count);
	// what the copy of three holds ago holds by now
	const unsigned old = (n + 1) & 3;
	if (t.number[old]) {
		const uint32_t *w = static_cast<const uint32_t *>(t.mapped[old]);
		uint32_t hash = 2166136261u;
		for (unsigned i = 0; i < strip_w * strip_h; i++) hash = (hash ^ w[i]) * 16777619u;
		RARCH_LOG("[FrameGen] trace frame %lu hash %08x\n", t.number[old], hash);
	}
	t.number[n & 3] = fg->unique_count;
	const VkBufferImageCopy region{ 0, 0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { int32_t(fg->width / 4), int32_t(fg->height / 2), 0 }, { strip_w, strip_h, 1 } };
	image_barrier(cmd, held.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
		VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	vkCmdCopyImageToBuffer(cmd, held.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.buffer[n & 3], 1, &region);
	image_barrier(cmd, held.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
		VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
	RARCH_LOG("[FrameGen] trace hold %lu image %llx view %llx had_held %d\n", fg->unique_count, (unsigned long long)cur->image,
		(unsigned long long)cur->view, int(fg->has_held));
}

void vk_framegen_hold(vk_framegen_t *fg, VkCommandBuffer cmd, unsigned slot, const struct vk_framegen_frame *cur)
{
	if (!fg || slot >= fg->slots || !ensure_resources(fg, cur)) {
		return;
	}
	const VkPipelineStageFlags compute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	Target &held = fg->frames[fg->held];
	// A copy through the "repeat" method, which converts any format to RGBA8. The image was last read by earlier
	// presents' generations, so the copy waits for them (layouts: sampled before, general to write, sampled after).
	image_barrier(cmd, held.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		compute | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, compute);
	const fg_frame_pair copy{ cur->view, cur->view, cur->view, cur->view, cur->view, 0.0f, cur->width, cur->height };
	query_begin(fg, cmd, slot, 1);
	fg_record_generate(fg->ctx, cmd, slot, FG_METHOD_REPEAT, &copy, held.view);
	query_end(fg, cmd, slot, 1);
	image_barrier(cmd, held.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
		VK_ACCESS_SHADER_READ_BIT, compute, compute | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
	const bool had_held = fg->has_held;
	fg->has_held = true;
	fg->tiles_found = false;
	fg->unique_count++;
	fg->latest = fg->held;
	fg->latest_valid = true;
	if (trace_enabled(fg)) {
		fg->has_held = had_held;
		trace_hold(fg, cmd, held, cur);
		fg->has_held = true;
	}
	if (!fg->dump.dir.empty()) {
		dump_hold(fg, cmd);
	}
}

bool vk_framegen_generate(vk_framegen_t *fg, VkCommandBuffer cmd, unsigned slot, float t, struct vk_framegen_frame *out)
{
	if (!fg || slot >= fg->slots || !fg->ctx || !fg->history_valid || !fg->has_held) {
		return false;
	}
	const VkPipelineStageFlags compute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	const VkPipelineStageFlags fragment = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	const Target &prev = fg->frames[fg->held ^ 1];
	const Target &next = fg->frames[fg->held];
	Target &gen = fg->generated[slot];
	// the generated image: written by the network, then sampled by the draw
	image_barrier(cmd, gen.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, compute);
	static const fg_method methods[] = { FG_METHOD_NEURAL, FG_METHOD_BLEND, FG_METHOD_REPEAT };
	// What is generated: the method chosen, or a cheaper one when the governor has stepped down. Pixel art goes with the
	// motion pass alone (it scored as well as with the network, in a third of the time; the network can be put back for tests
	// with PS5_FRAMEGEN_NETWORK_ON_PIXELS), and a repeat is its next step; others: the network, a blend, a repeat.
	unsigned method = fg->method % 3;
	uint32_t crisp = fg->pixel_art ? 1u : 0u;
	if (method == 0) {
		if (fg->pixel_art) {
			static const bool network = getenv("PS5_FRAMEGEN_NETWORK_ON_PIXELS") != nullptr;
			const unsigned step = fg->level + (network ? 0u : 1u);
			if (step == 1) {
				crisp = 5;
			} else if (step >= 2) {
				method = 2;
			}
		} else if (fg->level >= 1) {
			method = fg->level == 1 ? 1 : 2;
		}
	}
	const fg_frame_pair pair{ prev.view, next.view, prev.view, next.view, next.view, t, 0, 0, crisp, fg->tiles_found ? 1u : 0u };
	query_begin(fg, cmd, slot, 0);
	fg_record_generate(fg->ctx, cmd, slot, methods[method], &pair, gen.view);
	query_end(fg, cmd, slot, 0);
	fg->tiles_found = method == 0;
	fg->generated_count++;
	image_barrier(cmd, gen.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
		VK_ACCESS_SHADER_READ_BIT, compute, fragment | compute);
	if (out) {
		out->image = gen.image;
		out->view = gen.view;
		out->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		out->width = fg->width;
		out->height = fg->height;
		out->format = VK_FORMAT_R8G8B8A8_UNORM;
	}
	if (!fg->dump.dir.empty()) {
		dump_record(fg, cmd, true, gen, t);
	}
	fg->pic_generated = true;
	fg->pic_t = t;
	fg->presents++;
	return true;
}

bool vk_framegen_current(vk_framegen_t *fg, struct vk_framegen_frame *out)
{
	// Pixel art only: the held copy is the frame at 3 or 4 times its size, exactly (nearest), so showing it loses nothing
	if (!fg || !fg->ctx || !fg->pixel_art || !fg->latest_valid || !out) {
		return false;
	}
	const Target &t = fg->frames[fg->latest];
	out->image = t.image;
	out->view = t.view;
	out->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	out->width = fg->width;
	out->height = fg->height;
	out->format = VK_FORMAT_R8G8B8A8_UNORM;
	return true;
}

// Once for each new frame: the generator's GPU time against a picture's time
static void govern(vk_framegen *fg, double core_fps, double target_fps)
{
	if (target_fps > 1.0) fg->period_ms = 1000.0 / target_fps;
	if (fg->gen_ms <= 0.0) return;
	const double pictures = std::max(1.0, double(fg->times.size()));
	const double per_picture = fg->gen_ms + fg->hold_ms / pictures;
	static const double limit = getenv("PS5_FRAMEGEN_GOV_LIMIT") ? atof(getenv("PS5_FRAMEGEN_GOV_LIMIT")) : 0.4;   // (the share of a picture's time; for tests)
	if (per_picture > limit * fg->period_ms) {
		fg->under = 0;
		if (++fg->over >= 8 && fg->level < 2) {
			// Down again soon after going up: the level above is too dear for this game; wait longer before trying it again
			fg->up_after = fg->since_up < 3000 ? std::min(fg->up_after * 2, 20000u) : 600u;
			fg->level++;
			fg->over = 0;
			fg->gen_ms = fg->hold_ms = 0.0;
			RARCH_LOG("[FrameGen] governor: %.2f ms a picture is over %.2f; level %u\n", per_picture, limit * fg->period_ms, fg->level);
		}
	} else {
		fg->over = 0;
		if (fg->since_up < 1000000) fg->since_up++;
		if (fg->level > 0 && per_picture < 0.1 * fg->period_ms && ++fg->under >= fg->up_after) {
			fg->level--;
			fg->under = 0;
			fg->since_up = 0;
			fg->gen_ms = fg->hold_ms = 0.0;
			RARCH_LOG("[FrameGen] governor: room again; level %u\n", fg->level);
		}
	}
	(void)core_fps;
}

void vk_framegen_begin_group(vk_framegen_t *fg, unsigned calls, double gap_s, double core_fps, double target_fps, unsigned dup)
{
	if (!fg) return;
	calls = calls >= 1 ? calls : 1;
	if (!(gap_s > 0.0)) gap_s = double(calls) / (core_fps > 1.0 ? core_fps : 60.0);
	// another length: the pictures start afresh (a clock's jitter of a few percent is not another length)
	if (calls != fg->group_period || !(fg->group_gap_s > 0.0) || std::fabs(gap_s - fg->group_gap_s) > 0.05 * fg->group_gap_s) fg->phase = -1.0;
	fg->group_period = calls;
	fg->group_gap_s = gap_s;
	fg->nominal = (core_fps > 1.0 && target_fps > 1.0) ? std::max(1u, unsigned(target_fps / core_fps + 0.5)) : 1u;
	govern(fg, core_fps, target_fps);
	fg->dup = dup >= 1 ? dup : 1;
	fg->times.clear();
	// A game within 2% of a whole number of pictures per frame is taken as exactly that (the display is paced to it, as
	// for any game at the display's rate over a whole number): the pictures then fall at j/k with the real frame last
	if (target_fps > 1.0) {
		const double n = gap_s * target_fps, rr = std::floor(n + 0.5);
		if (rr >= 1.0 && std::fabs(n - rr) <= 0.02 * n) {
			gap_s = rr / target_fps;
		}
	}
	// A picture every 1/target seconds; a new-frame interval lasts gap_s seconds
	const double step = target_fps > 1.0 ? 1.0 / (target_fps * gap_s) : 1.0;
	if (!(fg->phase > 0.0) || fg->phase > 1.0 + step) fg->phase = std::min(1.0, step);
	while (fg->phase <= 1.0 + 1e-6 && fg->times.size() < 64) {
		fg->times.push_back(float(std::min(fg->phase, 1.0)));
		fg->phase += step;
	}
	fg->phase -= 1.0;
	if (fg->phase <= 0.0) fg->phase = step;
}

unsigned vk_framegen_call_presents(vk_framegen_t *fg, unsigned call)
{
	if (!fg) return 1;
	const unsigned p = fg->group_period, k = unsigned(fg->times.size());
	if (call >= p) return fg->nominal * fg->dup;   // beyond the predicted length: as many as a call is nominally worth, or the display would repeat
	const unsigned a = call * k / p, b = (call + 1) * k / p;
	return std::max(1u, b - a) * fg->dup;
}

bool vk_framegen_picture(vk_framegen_t *fg, unsigned call, unsigned present, unsigned presents_in_call, float *t, bool *final_present)
{
	if (!fg) return false;
	const unsigned p = fg->group_period, k = unsigned(fg->times.size());
	const unsigned a = call < p ? call * k / p : k;
	const unsigned pic = a + (present - 1) / fg->dup;
	if (final_present) *final_present = call + 1 >= p && present >= presents_in_call;
	if (pic >= k || fg->times[pic] >= 1.0f - 1e-4f) return false;
	*t = fg->times[pic];
	return true;
}

unsigned long vk_framegen_unique(const vk_framegen_t *fg)
{
	return fg ? fg->unique_count : 0;
}

// ---- the present-stream recorder ----

static std::atomic<unsigned> g_record_request{ 0 };

extern "C" void ps5_fg_record_request(unsigned pictures)
{
	g_record_request.store(pictures ? pictures : 240u);
}

static void record_free(vk_framegen *fg)
{
	Recorder &r = fg->rec;
	if (r.mapped) vkUnmapMemory(fg->device, r.buffer_memory);
	if (r.buffer) vkDestroyBuffer(fg->device, r.buffer, nullptr);
	if (r.buffer_memory) vkFreeMemory(fg->device, r.buffer_memory, nullptr);
	if (r.image) vkDestroyImage(fg->device, r.image, nullptr);
	if (r.image_memory) vkFreeMemory(fg->device, r.image_memory, nullptr);
	r.mapped = nullptr;
	r.buffer = VK_NULL_HANDLE;
	r.buffer_memory = VK_NULL_HANDLE;
	r.image = VK_NULL_HANDLE;
	r.image_memory = VK_NULL_HANDLE;
	r.active = false;
	r.meta.clear();
}

static bool record_begin(vk_framegen *fg, unsigned w, unsigned h, unsigned pictures)
{
	Recorder &r = fg->rec;
	static const unsigned want = getenv("PS5_FRAMEGEN_RECORD_WIDTH") ? unsigned(atoi(getenv("PS5_FRAMEGEN_RECORD_WIDTH"))) : 640u;
	r.width = std::min(std::max(want, 64u), w) & ~1u;
	r.height = std::max(2u, unsigned(double(h) * r.width / w + 0.5) & ~1u);
	r.capacity = pictures;
	r.count = 0;
	r.meta.clear();
	r.meta.reserve(pictures);
	const VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM,
		{ r.width, r.height, 1 }, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, VK_IMAGE_LAYOUT_UNDEFINED };
	VkPhysicalDeviceMemoryProperties mem;
	vkGetPhysicalDeviceMemoryProperties(fg->gpu, &mem);
	VkMemoryRequirements req;
	if (vkCreateImage(fg->device, &imageInfo, nullptr, &r.image) != VK_SUCCESS) goto fail;
	vkGetImageMemoryRequirements(fg->device, r.image, &req);
	{
		const uint32_t type = find_memory(mem, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		if (type == UINT32_MAX) goto fail;
		const VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, type };
		if (vkAllocateMemory(fg->device, &ai, nullptr, &r.image_memory) != VK_SUCCESS) goto fail;
		if (vkBindImageMemory(fg->device, r.image, r.image_memory, 0) != VK_SUCCESS) goto fail;
	}
	{
		VkBufferCreateInfo bi{};
		bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bi.size = VkDeviceSize(r.width) * r.height * 4 * pictures;
		bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		if (vkCreateBuffer(fg->device, &bi, nullptr, &r.buffer) != VK_SUCCESS) goto fail;
		vkGetBufferMemoryRequirements(fg->device, r.buffer, &req);
		uint32_t type = find_memory(mem, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
		if (type == UINT32_MAX) type = find_memory(mem, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		if (type == UINT32_MAX) goto fail;
		const VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, type };
		if (vkAllocateMemory(fg->device, &ai, nullptr, &r.buffer_memory) != VK_SUCCESS) goto fail;
		if (vkBindBufferMemory(fg->device, r.buffer, r.buffer_memory, 0) != VK_SUCCESS) goto fail;
		void *ptr = nullptr;
		if (vkMapMemory(fg->device, r.buffer_memory, 0, VK_WHOLE_SIZE, 0, &ptr) != VK_SUCCESS) goto fail;
		r.mapped = static_cast<unsigned char *>(ptr);
	}
	r.active = true;
	RARCH_LOG("[FrameGen] record: %u pictures of %ux%u (from %ux%u)\n", pictures, r.width, r.height, w, h);
	return true;
fail:
	RARCH_ERR("[FrameGen] record: no memory for %u pictures of %ux%u; not recording\n", pictures, r.width, r.height);
	record_free(fg);
	return false;
}

static void record_write(vk_framegen *fg)
{
	Recorder &r = fg->rec;
	vkDeviceWaitIdle(fg->device);
	char folder[512];
	snprintf(folder, sizeof(folder), "%s/rec-%lu", r.dir.c_str(), ++r.runs);
	mkdir(r.dir.c_str(), 0755);
	mkdir(folder, 0755);
	char path[600];
	snprintf(path, sizeof(path), "%s/frames.rgba", folder);
	if (FILE *f = fopen(path, "wb")) {
		fwrite(r.mapped, 1, size_t(r.width) * r.height * 4 * r.count, f);
		fclose(f);
	}
	snprintf(path, sizeof(path), "%s/size.txt", folder);
	if (FILE *f = fopen(path, "w")) {
		fprintf(f, "%u %u %u\n", r.width, r.height, r.count);
		fclose(f);
	}
	snprintf(path, sizeof(path), "%s/index.txt", folder);
	if (FILE *f = fopen(path, "w")) {
		fprintf(f, "# i place ns generated t unique level\n");
		for (unsigned i = 0; i < r.meta.size(); i++) {
			const Recorder::Meta &m = r.meta[i];
			fprintf(f, "%u %lu %llu %d %.5f %lu %u\n", i, m.place, m.ns, m.generated ? 1 : 0, m.t, m.unique, m.level);
		}
		fclose(f);
	}
	RARCH_LOG("[FrameGen] record: %u pictures written to %s\n", r.count, folder);
	record_free(fg);
}

void vk_framegen_record(vk_framegen_t *fg, VkCommandBuffer cmd, VkImage backbuffer, unsigned w, unsigned h)
{
	if (!fg || !backbuffer || w == 0 || h == 0) return;
	Recorder &r = fg->rec;
	const bool generated = fg->pic_generated;
	const float t = generated ? fg->pic_t : 1.0f;
	fg->pic_generated = false;
	r.serial++;
	if (r.active && r.count >= r.capacity) {
		record_write(fg);   // (the last picture's copy was submitted with an earlier command buffer)
	}
	if (!r.active) {
		unsigned pictures = g_record_request.exchange(0);
		if (!pictures && r.env_at && !r.env_done && r.serial >= r.env_at) {
			pictures = r.env_count;
			r.env_done = true;
			if (r.dir.empty()) return;
		}
		if (!pictures) return;
		if (r.dir.empty()) r.dir = "/app0";
		if (!record_begin(fg, w, h, std::min(pictures, 2000u))) return;
	}
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	const VkImageBlit blit{ { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { { 0, 0, 0 }, { int32_t(w), int32_t(h), 1 } },
		{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { { 0, 0, 0 }, { int32_t(r.width), int32_t(r.height), 1 } } };
	const VkBufferImageCopy region{ VkDeviceSize(r.count) * r.width * r.height * 4, 0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { r.width, r.height, 1 } };
	// (the swapchain image is in the colour attachment layout after the render pass, and goes back to it)
	image_barrier(cmd, backbuffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	image_barrier(cmd, r.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	vkCmdBlitImage(cmd, backbuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, r.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
	image_barrier(cmd, r.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	vkCmdCopyImageToBuffer(cmd, r.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, r.buffer, 1, &region);
	image_barrier(cmd, backbuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
		VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
	r.meta.push_back({ r.serial, (unsigned long long)now.tv_sec * 1000000000ull + (unsigned long long)now.tv_nsec, generated, t, fg->unique_count, fg->level });
	r.count++;
}

void vk_framegen_commit(vk_framegen_t *fg)
{
	if (!fg || !fg->has_held) {
		return;
	}
	if (fg->trace.enabled == 1) RARCH_LOG("[FrameGen] trace commit after hold %lu\n", fg->unique_count);
	fg->held ^= 1;            // the held frame is the previous one, and the other image is free to hold the next
	fg->has_held = false;
	fg->history_valid = true;
}
