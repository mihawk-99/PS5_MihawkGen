/*
 * PS5_FrameGen - runs the neural method on a PC's Vulkan device, headless: two RGBA8
 * frames and a weight file in, the generated frame out. train/parity.py compares it with
 * the network in PyTorch; the same program times the passes (GPU timestamps).
 *
 *   neural_run WEIGHTS FRAME0.rgba FRAME1.rgba WIDTH HEIGHT T OUTPUT.rgba [REPEATS]
 *
 * Copyright (C) 2026 Mihawk
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

#include "framegen.h"

#define VK(call)                                                          \
	do {                                                                  \
		VkResult r_ = (call);                                             \
		if (r_ != VK_SUCCESS) {                                           \
			std::fprintf(stderr, "%s failed: %d\n", #call, int(r_));      \
			std::exit(1);                                                 \
		}                                                                 \
	} while (0)

static std::vector<char> slurp(const char *path)
{
	std::ifstream f(path, std::ios::binary);
	if (!f) {
		std::fprintf(stderr, "cannot read %s\n", path);
		std::exit(1);
	}
	return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char **argv)
{
	if (argc < 8) {
		std::fprintf(stderr, "usage: %s WEIGHTS FRAME0.rgba FRAME1.rgba WIDTH HEIGHT T OUTPUT.rgba [REPEATS]\n", argv[0]);
		return 2;
	}
	const uint32_t width = uint32_t(atoi(argv[4])), height = uint32_t(atoi(argv[5]));
	const float t = float(atof(argv[6]));
	const int repeats = argc > 8 ? atoi(argv[8]) : 1;
	const std::vector<char> weights = slurp(argv[1]);
	const std::vector<char> frames[2] = { slurp(argv[2]), slurp(argv[3]) };
	const size_t bytes = size_t(width) * height * 4;
	if (frames[0].size() != bytes || frames[1].size() != bytes) {
		std::fprintf(stderr, "frames must be %zu bytes\n", bytes);
		return 2;
	}

	const VkApplicationInfo app{ .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3 };
	const VkInstanceCreateInfo instInfo{ .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
	VkInstance instance;
	VK(vkCreateInstance(&instInfo, nullptr, &instance));
	uint32_t count = 0;
	VK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
	std::vector<VkPhysicalDevice> devices(count);
	VK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
	VkPhysicalDevice physical = devices[0];
	const char *want = std::getenv("FG_DEVICE");
	for (VkPhysicalDevice dev : devices) {
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(dev, &props);
		if (want && std::strstr(props.deviceName, want)) {
			physical = dev;
		}
	}
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(physical, &props);
	std::fprintf(stderr, "device: %s\n", props.deviceName);
	uint32_t family = 0;
	uint32_t qcount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physical, &qcount, nullptr);
	std::vector<VkQueueFamilyProperties> families(qcount);
	vkGetPhysicalDeviceQueueFamilyProperties(physical, &qcount, families.data());
	for (uint32_t i = 0; i < qcount; i++) {
		if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = i; break; }
	}
	const float priority = 1.0f;
	const VkDeviceQueueCreateInfo queueInfo{ .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority };
	const VkDeviceCreateInfo devInfo{ .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &queueInfo };
	VkDevice device;
	VK(vkCreateDevice(physical, &devInfo, nullptr, &device));
	VkQueue queue;
	vkGetDeviceQueue(device, family, 0, &queue);
	VkPhysicalDeviceMemoryProperties memory;
	vkGetPhysicalDeviceMemoryProperties(physical, &memory);

	auto findMemory = [&](uint32_t bits, VkMemoryPropertyFlags flags) {
		for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
			if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & flags) == flags) return i;
		}
		std::fprintf(stderr, "no memory type\n");
		std::exit(1);
	};
	struct Img { VkImage image; VkDeviceMemory mem; VkImageView view; };
	auto makeImage = [&](VkImageUsageFlags usage) {
		Img img{};
		const VkImageCreateInfo info{ .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
			.extent = { width, height, 1 }, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
		VK(vkCreateImage(device, &info, nullptr, &img.image));
		VkMemoryRequirements req;
		vkGetImageMemoryRequirements(device, img.image, &req);
		const VkMemoryAllocateInfo ai{ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = findMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
		VK(vkAllocateMemory(device, &ai, nullptr, &img.mem));
		VK(vkBindImageMemory(device, img.image, img.mem, 0));
		const VkImageViewCreateInfo vi{ .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img.image, .viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = VK_FORMAT_R8G8B8A8_UNORM, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
		VK(vkCreateImageView(device, &vi, nullptr, &img.view));
		return img;
	};
	struct Buf { VkBuffer buffer; VkDeviceMemory mem; void *mapped; };
	auto makeBuffer = [&](size_t size) {
		Buf b{};
		const VkBufferCreateInfo info{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
			.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT };
		VK(vkCreateBuffer(device, &info, nullptr, &b.buffer));
		VkMemoryRequirements req;
		vkGetBufferMemoryRequirements(device, b.buffer, &req);
		const VkMemoryAllocateInfo ai{ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
			.memoryTypeIndex = findMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
		VK(vkAllocateMemory(device, &ai, nullptr, &b.mem));
		VK(vkBindBufferMemory(device, b.buffer, b.mem, 0));
		VK(vkMapMemory(device, b.mem, 0, VK_WHOLE_SIZE, 0, &b.mapped));
		return b;
	};
	Img in[2] = { makeImage(VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT), makeImage(VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT) };
	Img out = makeImage(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
	Buf staging = makeBuffer(bytes);

	const fg_create_info info{ .device = device, .physical_device = physical, .width = width, .height = height, .slots = 1,
		.output_format = VK_FORMAT_R8G8B8A8_UNORM,
		.source_width = getenv("FG_SOURCE") ? uint32_t(atoi(getenv("FG_SOURCE"))) : 0u,       // pixel art: the source size ("W,H")
		.source_height = getenv("FG_SOURCE") && strchr(getenv("FG_SOURCE"), ',') ? uint32_t(atoi(strchr(getenv("FG_SOURCE"), ',') + 1)) : 0u };
	fg_context *ctx = nullptr;
	char error[256];
	VK(fg_context_create(&info, &ctx, error, sizeof(error)));
	if (fg_context_load_network(ctx, weights.data(), weights.size(), error, sizeof(error)) != VK_SUCCESS) {
		std::fprintf(stderr, "%s\n", error);
		return 1;
	}

	const VkCommandPoolCreateInfo poolInfo{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = family };
	VkCommandPool pool;
	VK(vkCreateCommandPool(device, &poolInfo, nullptr, &pool));
	auto submit = [&](auto record) {
		VkCommandBuffer cmd;
		const VkCommandBufferAllocateInfo ai{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
		VK(vkAllocateCommandBuffers(device, &ai, &cmd));
		const VkCommandBufferBeginInfo bi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
		VK(vkBeginCommandBuffer(cmd, &bi));
		record(cmd);
		VK(vkEndCommandBuffer(cmd));
		const VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
		VK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
		VK(vkQueueWaitIdle(queue));
		vkFreeCommandBuffers(device, pool, 1, &cmd);
	};
	auto barrier = [&](VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags srcA, VkAccessFlags dstA,
		VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
		const VkImageMemoryBarrier b{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = srcA, .dstAccessMask = dstA, .oldLayout = from,
			.newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = image,
			.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
		vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
	};
	const VkBufferImageCopy region{ .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { width, height, 1 } };
	for (int f = 0; f < 2; f++) {
		std::memcpy(staging.mapped, frames[f].data(), bytes);
		submit([&](VkCommandBuffer cmd) {
			barrier(cmd, in[f].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
			vkCmdCopyBufferToImage(cmd, staging.buffer, in[f].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
			barrier(cmd, in[f].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		});
	}
	submit([&](VkCommandBuffer cmd) {
		barrier(cmd, out.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	});

	const fg_frame_pair pair{ .prev_color = in[0].view, .next_color = in[1].view, .prev_depth = VK_NULL_HANDLE, .next_depth = VK_NULL_HANDLE,
		.next_motion = VK_NULL_HANDLE, .t = t };
	VkQueryPool timestamps;
	const VkQueryPoolCreateInfo qi{ .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 };
	VK(vkCreateQueryPool(device, &qi, nullptr, &timestamps));
	double total_ms = 0.0;
	for (int r = 0; r < repeats; r++) {
		submit([&](VkCommandBuffer cmd) {
			vkCmdResetQueryPool(cmd, timestamps, 0, 2);
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 0);
			fg_record_generate(ctx, cmd, 0, FG_METHOD_NEURAL, &pair, out.view);
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 1);
		});
		uint64_t ticks[2];
		VK(vkGetQueryPoolResults(device, timestamps, 0, 2, sizeof(ticks), ticks, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));
		const double ms = double(ticks[1] - ticks[0]) * props.limits.timestampPeriod / 1e6;
		if (r > 0 || repeats == 1) total_ms += ms;
		if (r == repeats - 1) std::fprintf(stderr, "last run: %.3f ms\n", ms);
	}
	std::fprintf(stderr, "mean over %d runs (first excluded when more than one): %.3f ms\n", repeats, total_ms / (repeats > 1 ? repeats - 1 : 1));

	submit([&](VkCommandBuffer cmd) {
		barrier(cmd, out.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
		vkCmdCopyImageToBuffer(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.buffer, 1, &region);
	});
	std::ofstream(argv[7], std::ios::binary).write(static_cast<const char *>(staging.mapped), std::streamsize(bytes));
	fg_context_destroy(ctx);
	vkDeviceWaitIdle(device);
	return 0;
}
