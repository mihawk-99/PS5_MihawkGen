/*
 * PS5_FrameGen - runs the motion-vector method (FG_METHOD_MVWARP) on a PC's Vulkan device, headless, on a list of jobs: two RGBA8
 * frames, the next frame's motion vectors (float32 pairs in pixels, in FidelityFX's convention: where the pixel was in the
 * previous frame minus where it is) and its depth (float32); the frame at t = 1/2 comes out. It is for comparing the methods with
 * FSR 3.1 on the same exact scene.
 *
 *   [METHOD=mvwarp|mvsplat] mvwarp_run WIDTH HEIGHT JOBS.txt     JOBS: lines "frame0 frame1 mv depth out [prev_depth]"
 *   (the depth of the previous frame is the next frame's when the line has no sixth word; depths are linear, smaller nearer)
 *
 * Copyright (C) 2026 Mihawk
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
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

static std::vector<char> slurp(const std::string &path)
{
	std::ifstream f(path, std::ios::binary);
	if (!f) {
		std::fprintf(stderr, "cannot read %s\n", path.c_str());
		std::exit(1);
	}
	return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static uint16_t to_half(float v)
{
	uint32_t x;
	std::memcpy(&x, &v, 4);
	const uint32_t s = (x >> 16) & 0x8000;
	const int e = int((x >> 23) & 255) - 127 + 15;
	const uint32_t m = x & 0x7fffff;
	return e <= 0 ? uint16_t(s) : e >= 31 ? uint16_t(s | 0x7bff) : uint16_t(s | (e << 10) | (m >> 13));
}

int main(int argc, char **argv)
{
	if (argc < 4) {
		std::fprintf(stderr, "usage: %s WIDTH HEIGHT JOBS.txt\n", argv[0]);
		return 2;
	}
	const uint32_t width = uint32_t(atoi(argv[1])), height = uint32_t(atoi(argv[2]));
	const VkApplicationInfo app{ .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3 };
	const VkInstanceCreateInfo instInfo{ .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
	VkInstance instance;
	VK(vkCreateInstance(&instInfo, nullptr, &instance));
	uint32_t count = 0;
	VK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
	std::vector<VkPhysicalDevice> devices(count);
	VK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
	VkPhysicalDevice physical = devices[0];
	uint32_t family = 0, qcount = 0;
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
	auto makeImage = [&](VkFormat format, VkImageUsageFlags usage) {
		Img img{};
		const VkImageCreateInfo info{ .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = format,
			.extent = { width, height, 1 }, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
		VK(vkCreateImage(device, &info, nullptr, &img.image));
		VkMemoryRequirements req;
		vkGetImageMemoryRequirements(device, img.image, &req);
		const VkMemoryAllocateInfo ai{ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = findMemory(req.memoryTypeBits, 0) };
		VK(vkAllocateMemory(device, &ai, nullptr, &img.mem));
		VK(vkBindImageMemory(device, img.image, img.mem, 0));
		const VkImageViewCreateInfo vi{ .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img.image, .viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = format, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
		VK(vkCreateImageView(device, &vi, nullptr, &img.view));
		return img;
	};
	const size_t bytes = size_t(width) * height * 8;   // the largest upload: RG16F is 4, RGBA8 4, R32F 4
	VkBuffer buffer;
	VkDeviceMemory bufferMem;
	void *mapped;
	{
		const VkBufferCreateInfo info{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT };
		VK(vkCreateBuffer(device, &info, nullptr, &buffer));
		VkMemoryRequirements req;
		vkGetBufferMemoryRequirements(device, buffer, &req);
		const VkMemoryAllocateInfo ai{ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
			.memoryTypeIndex = findMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
		VK(vkAllocateMemory(device, &ai, nullptr, &bufferMem));
		VK(vkBindBufferMemory(device, buffer, bufferMem, 0));
		VK(vkMapMemory(device, bufferMem, 0, VK_WHOLE_SIZE, 0, &mapped));
	}
	const VkImageUsageFlags sampled = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	Img color[2] = { makeImage(VK_FORMAT_R8G8B8A8_UNORM, sampled), makeImage(VK_FORMAT_R8G8B8A8_UNORM, sampled) };
	Img depth[2] = { makeImage(VK_FORMAT_R32_SFLOAT, sampled), makeImage(VK_FORMAT_R32_SFLOAT, sampled) };
	Img motion = makeImage(VK_FORMAT_R16G16_SFLOAT, sampled);
	Img out = makeImage(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

	const fg_create_info info{ .device = device, .physical_device = physical, .width = width, .height = height, .slots = 1, .output_format = VK_FORMAT_R8G8B8A8_UNORM };
	fg_context *ctx = nullptr;
	char error[256];
	VK(fg_context_create(&info, &ctx, error, sizeof(error)));
	const char *methodName = std::getenv("METHOD");
	const fg_method method = methodName && std::strcmp(methodName, "mvsplat") == 0 ? FG_METHOD_MVSPLAT : FG_METHOD_MVWARP;
	if (method == FG_METHOD_MVSPLAT && fg_context_enable_mvsplat(ctx, error, sizeof(error)) != VK_SUCCESS) {
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
	auto barrier = [&](VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to) {
		const VkImageMemoryBarrier b{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
			.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, .oldLayout = from, .newLayout = to,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = image, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
	};
	const VkBufferImageCopy region{ .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { width, height, 1 } };
	auto upload = [&](Img &img, const void *data, size_t n) {
		std::memcpy(mapped, data, n);
		submit([&](VkCommandBuffer cmd) {
			barrier(cmd, img.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
			vkCmdCopyBufferToImage(cmd, buffer, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
			barrier(cmd, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		});
	};
	submit([&](VkCommandBuffer cmd) { barrier(cmd, out.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL); });

	std::ifstream jobs(argv[3]);
	std::string f0, f1, mvf, df, of, line;
	int done = 0;
	while (std::getline(jobs, line)) {
		std::string pdf;
		{
			std::istringstream words(line);
			if (!(words >> f0 >> f1 >> mvf >> df >> of)) continue;
			words >> pdf;
		}
		upload(color[0], slurp(f0).data(), bytes / 2);
		upload(color[1], slurp(f1).data(), bytes / 2);
		// the file has the next frame's pixels' (previous position - position); the method wants (position - previous position) in uv
		const std::vector<char> mvraw = slurp(mvf);
		const float *mv = reinterpret_cast<const float *>(mvraw.data());
		std::vector<uint16_t> half(size_t(width) * height * 2);
		for (size_t i = 0; i < size_t(width) * height; i++) {
			half[2 * i] = to_half(-mv[2 * i] / float(width));
			half[2 * i + 1] = to_half(-mv[2 * i + 1] / float(height));
		}
		upload(motion, half.data(), half.size() * 2);
		const std::vector<char> dd = slurp(df);
		const std::vector<char> pd = pdf.empty() ? dd : slurp(pdf);
		upload(depth[0], pd.data(), pd.size());
		upload(depth[1], dd.data(), dd.size());
		const fg_frame_pair pair{ .prev_color = color[0].view, .next_color = color[1].view, .prev_depth = depth[0].view, .next_depth = depth[1].view,
			.next_motion = motion.view, .t = 0.5f };
		submit([&](VkCommandBuffer cmd) { fg_record_generate(ctx, cmd, 0, method, &pair, out.view); });
		submit([&](VkCommandBuffer cmd) {
			barrier(cmd, out.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
			vkCmdCopyImageToBuffer(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
			barrier(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
		});
		std::ofstream(of, std::ios::binary).write(static_cast<const char *>(mapped), std::streamsize(bytes / 2));
		done++;
	}
	std::fprintf(stderr, "jobs: %d\n", done);
	fg_context_destroy(ctx);
	vkDeviceWaitIdle(device);
	return 0;
}
