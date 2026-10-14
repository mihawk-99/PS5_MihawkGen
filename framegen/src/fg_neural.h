/*
 * PS5_FrameGen - the neural method's interface to the context (fg_neural.cpp).
 *
 * Copyright (C) 2026 Mihawk
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include "framegen.h"

struct fg_neural;

struct fg_neural_info {
	VkDevice device;
	VkPhysicalDevice physical_device;
	uint32_t width, height, slots;
	uint32_t source_width, source_height; // pixel art: the frames in source pixels (0: no motion pass)
	/* UNDEFINED keeps the original RGBA8 output. Also supports A2B10G10R10_UNORM_PACK32
	 * and R16G16B16A16_SFLOAT; the caller supplies a matching storage view. */
	VkFormat output_format;
	/* PQ/BT.2020 inputs and output, including PQ (not scRGB) in RGBA16F storage.
	 * Pixel-art source sizes are not supported; crisp is ignored. False preserves SDR. */
	bool hdr10;
	/* Required, from the caller's Vulkan implementation. No global loader is used. */
	PFN_vkGetDeviceProcAddr get_device_proc_addr;
	PFN_vkGetPhysicalDeviceMemoryProperties get_memory_properties;
	/* Optional Vulkan host allocation callbacks, copied on create. pUserData must outlive the instance. */
	const VkAllocationCallbacks *allocator;
};

/* Parses the weight file (train/export.py's format) and builds the pipelines, images and
 * descriptor sets. nullptr on failure, with `error` saying why. */
fg_neural *fg_neural_create(const fg_neural_info *info, const void *blob, size_t size, char *error, size_t error_size);
void fg_neural_destroy(fg_neural *neural);
/* Records the method's passes into `cmd`, with the barriers between them. */
void fg_neural_record(fg_neural *neural, VkCommandBuffer cmd, uint32_t slot, const fg_frame_pair *pair, VkImageView output);

/* Profiling: see fg_context_set_profile */
void fg_neural_set_profile(fg_neural *neural, VkQueryPool pool, uint32_t base);
uint32_t fg_neural_pass_count(const fg_neural *neural);
const char *fg_neural_pass_label(const fg_neural *neural, uint32_t pass);

/* Device-local entry points shared by the context and the standalone neural backend. */
#define FG_DEVICE_FUNCTIONS(X) \
	X(AllocateDescriptorSets) \
	X(AllocateMemory) \
	X(BindBufferMemory) \
	X(BindImageMemory) \
	X(CmdBindDescriptorSets) \
	X(CmdBindPipeline) \
	X(CmdClearColorImage) \
	X(CmdDispatch) \
	X(CmdPipelineBarrier) \
	X(CmdPushConstants) \
	X(CmdWriteTimestamp) \
	X(CreateBuffer) \
	X(CreateComputePipelines) \
	X(CreateDescriptorPool) \
	X(CreateDescriptorSetLayout) \
	X(CreateImage) \
	X(CreateImageView) \
	X(CreatePipelineLayout) \
	X(CreateSampler) \
	X(CreateShaderModule) \
	X(DestroyBuffer) \
	X(DestroyDescriptorPool) \
	X(DestroyDescriptorSetLayout) \
	X(DestroyImage) \
	X(DestroyImageView) \
	X(DestroyPipeline) \
	X(DestroyPipelineLayout) \
	X(DestroySampler) \
	X(DestroyShaderModule) \
	X(FreeMemory) \
	X(GetBufferMemoryRequirements) \
	X(GetImageMemoryRequirements) \
	X(MapMemory) \
	X(UnmapMemory) \
	X(UpdateDescriptorSets)

struct fg_device_dispatch {
#define FG_DECLARE(name) PFN_vk##name name{};
	FG_DEVICE_FUNCTIONS(FG_DECLARE)
#undef FG_DECLARE

	/* Returns the first unavailable entry point, or nullptr on success. */
	const char *load(VkDevice device, PFN_vkGetDeviceProcAddr get)
	{
#define FG_LOAD(name) \
		name = reinterpret_cast<PFN_vk##name>(get(device, "vk" #name)); \
		if (!name) return "vk" #name;
		FG_DEVICE_FUNCTIONS(FG_LOAD)
#undef FG_LOAD
		return nullptr;
	}
};
#undef FG_DEVICE_FUNCTIONS
