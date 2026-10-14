/*
 * PS5_FrameGen - host tests of the generator's device-free parts, and of the
 * SPIR-V it carries (run.sh builds and runs them; no GPU needed).
 *
 * Copyright (C) 2026 Mihawk
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "framegen.h"
#include "fg_neural.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

#include "shaders/blend.comp.h"
#include "shaders/mvwarp.comp.h"
#include "shaders/sp_clear.comp.h"
#include "shaders/sp_depth.comp.h"
#include "shaders/sp_down.comp.h"
#include "shaders/sp_fill.comp.h"
#include "shaders/sp_final.comp.h"
#include "shaders/sp_gather.comp.h"
#include "shaders/sp_up.comp.h"
#include "shaders/sp_vec.comp.h"
#include "shaders/nn_conv.comp.h"
#include "shaders/nn_downsample.comp.h"
#include "shaders/nn_final.comp.h"
#include "shaders/nn_pool.comp.h"
#include "shaders/nn_prep.comp.h"
#include "shaders/nn_update.comp.h"
#include "shaders/repeat.comp.h"

static int checks = 0;
static int failures = 0;

#define CHECK(condition)                                                          \
	do {                                                                          \
		checks++;                                                                 \
		if (!(condition)) {                                                       \
			failures++;                                                           \
			std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
		}                                                                         \
	} while (0)

static fg_create_info valid_info()
{
	fg_create_info info{};
	info.device = reinterpret_cast<VkDevice>(static_cast<uintptr_t>(1)); // never called through
	info.width = 3840;
	info.height = 2160;
	info.slots = 2;
	info.output_format = VK_FORMAT_R8G8B8A8_UNORM;
	info.get_device_proc_addr = vkGetDeviceProcAddr;
	info.get_memory_properties = vkGetPhysicalDeviceMemoryProperties;
	return info;
}

static void test_method_names()
{
	std::set<std::string> names;
	for (uint32_t m = 0; m < FG_METHOD_COUNT; m++) {
		const std::string name = fg_method_name(static_cast<fg_method>(m));
		CHECK(name != "unknown");
		CHECK(!name.empty());
		names.insert(name);
	}
	CHECK(names.size() == FG_METHOD_COUNT);
	CHECK(std::string(fg_method_name(FG_METHOD_COUNT)) == "unknown");
}

static void test_create_info()
{
	fg_create_info info = valid_info();
	CHECK(fg_check_create_info(&info) == nullptr);
	CHECK(fg_check_create_info(nullptr) != nullptr);
	info = valid_info(); info.device = VK_NULL_HANDLE; CHECK(fg_check_create_info(&info) != nullptr);
	info = valid_info(); info.width = 0; CHECK(fg_check_create_info(&info) != nullptr);
	info = valid_info(); info.height = 16385; CHECK(fg_check_create_info(&info) != nullptr);
	info = valid_info(); info.slots = 0; CHECK(fg_check_create_info(&info) != nullptr);
	info = valid_info(); info.slots = 9; CHECK(fg_check_create_info(&info) != nullptr);
	info = valid_info(); info.slots = 8; CHECK(fg_check_create_info(&info) == nullptr);
	info = valid_info(); info.output_format = VK_FORMAT_B8G8R8A8_UNORM; CHECK(fg_check_create_info(&info) != nullptr);
	// fg_context_create refuses the same, without touching the device
	info = valid_info(); info.slots = 0;
	fg_context *ctx = reinterpret_cast<fg_context *>(static_cast<uintptr_t>(1));
	char error[128] = "";
	CHECK(fg_context_create(&info, &ctx, error, sizeof(error)) != VK_SUCCESS);
	CHECK(ctx == nullptr);
	CHECK(std::strstr(error, "slots") != nullptr);
}

static PFN_vkVoidFunction VKAPI_CALL missing_device_proc(VkDevice, const char *)
{
	return nullptr;
}

static std::vector<uint32_t> test_weights()
{
	// One scale-1 block, width 4, two hidden layers, radius 0; all weights are zero.
	std::vector<uint32_t> weights{ 0x31574746, 1, 1, 4, 2, 0 };
	const uint32_t layers[][3] = { { 8, 4, 1 }, { 4, 4, 1 }, { 4, 5, 0 } };
	for (const auto &layer : layers) {
		weights.insert(weights.end(), layer, layer + 3);
		const uint32_t ig = (layer[0] + 3) / 4, og = (layer[1] + 3) / 4;
		weights.resize(weights.size() + og * 9 * ig * 16 + og * 4);
	}
	return weights;
}

static void test_external_dispatch()
{
	fg_create_info info = valid_info();
	info.get_device_proc_addr = nullptr;
	fg_context *ctx = nullptr;
	char error[256] = "";
	CHECK(fg_context_create(&info, &ctx, error, sizeof(error)) != VK_SUCCESS);
	CHECK(ctx == nullptr);
	CHECK(std::strstr(error, "get_device_proc_addr") != nullptr);
	info.get_device_proc_addr = missing_device_proc;
	CHECK(fg_context_create(&info, &ctx, error, sizeof(error)) != VK_SUCCESS);
	CHECK(ctx == nullptr);
	CHECK(std::strstr(error, "vkAllocateDescriptorSets") != nullptr);

	const std::vector<uint32_t> weights = test_weights();
	const size_t bytes = weights.size() * sizeof(uint32_t);
	fg_neural_info neural{};
	neural.device = info.device;
	neural.physical_device = reinterpret_cast<VkPhysicalDevice>(static_cast<uintptr_t>(1));
	neural.width = neural.height = 64;
	neural.slots = 1;
	neural.get_memory_properties = vkGetPhysicalDeviceMemoryProperties;
	CHECK(fg_neural_create(&neural, weights.data(), bytes, error, sizeof(error)) == nullptr);
	CHECK(std::strstr(error, "get_device_proc_addr") != nullptr);
	neural.get_device_proc_addr = missing_device_proc;
	neural.output_format = VK_FORMAT_B8G8R8A8_UNORM;
	CHECK(fg_neural_create(&neural, weights.data(), bytes, error, sizeof(error)) == nullptr);
	CHECK(std::strstr(error, "output format") != nullptr);
	neural.output_format = VK_FORMAT_UNDEFINED;
	neural.hdr10 = true;
	neural.source_width = neural.source_height = 32;
	CHECK(fg_neural_create(&neural, weights.data(), bytes, error, sizeof(error)) == nullptr);
	CHECK(std::strstr(error, "pixel art") != nullptr);
	neural.source_width = neural.source_height = 0;
	CHECK(fg_neural_create(&neural, weights.data(), bytes, error, sizeof(error)) == nullptr);
	CHECK(std::strstr(error, "vkAllocateDescriptorSets") != nullptr);
	CHECK(fg_neural_create(nullptr, weights.data(), bytes, error, sizeof(error)) == nullptr);
	CHECK(fg_neural_create(&neural, nullptr, bytes, error, sizeof(error)) == nullptr);
	neural.width = 0;
	CHECK(fg_neural_create(&neural, weights.data(), bytes, error, sizeof(error)) == nullptr);
	neural.width = 64;
	neural.slots = 9;
	CHECK(fg_neural_create(&neural, weights.data(), bytes, error, sizeof(error)) == nullptr);
	neural.slots = 1;
	neural.get_memory_properties = nullptr;
	CHECK(fg_neural_create(&neural, weights.data(), bytes, error, sizeof(error)) == nullptr);
	CHECK(std::strstr(error, "get_memory_properties") != nullptr);
	neural.get_memory_properties = vkGetPhysicalDeviceMemoryProperties;
	CHECK(fg_neural_create(&neural, weights.data(), 0, error, sizeof(error)) == nullptr);
	CHECK(std::strstr(error, "magic") != nullptr);
	const uint32_t empty_weights[] = { 0x31574746, 0 };
	CHECK(fg_neural_create(&neural, empty_weights, sizeof(empty_weights), error, sizeof(error)) == nullptr);
	CHECK(std::strstr(error, "blocks") != nullptr);
	CHECK(fg_neural_create(&neural, weights.data(), bytes - 4, error, sizeof(error)) == nullptr);
	CHECK(std::strstr(error, "truncated") != nullptr);
}

static void test_psnr()
{
	CHECK(std::isinf(fg_psnr(0.0, 100)));
	CHECK(std::isnan(fg_psnr(1.0, 0)));
	// mse 1e-4: 40 dB; mse 1e-2: 20 dB; mse 1: 0 dB
	CHECK(std::fabs(fg_psnr(1e-4 * 1000.0, 1000) - 40.0) < 1e-9);
	CHECK(std::fabs(fg_psnr(1e-2 * 300.0, 300) - 20.0) < 1e-9);
	CHECK(std::fabs(fg_psnr(5.0, 5)) < 1e-12);
}

/* The arrays compiled into the library are the .spv files beside the GLSL:
 * catches a header left stale after a shader change. */
static void test_embedded_spirv(const char *shader_dir)
{
	struct Embedded { const char *name; const uint32_t *words; size_t bytes; };
	const Embedded embedded[] = {
		{ "repeat", fg_spv_repeat, sizeof(fg_spv_repeat) },
		{ "blend", fg_spv_blend, sizeof(fg_spv_blend) },
		{ "mvwarp", fg_spv_mvwarp, sizeof(fg_spv_mvwarp) },
		{ "nn_downsample", fg_spv_nn_downsample, sizeof(fg_spv_nn_downsample) },
		{ "nn_pool", fg_spv_nn_pool, sizeof(fg_spv_nn_pool) },
		{ "nn_prep", fg_spv_nn_prep, sizeof(fg_spv_nn_prep) },
		{ "nn_conv", fg_spv_nn_conv, sizeof(fg_spv_nn_conv) },
		{ "nn_update", fg_spv_nn_update, sizeof(fg_spv_nn_update) },
		{ "nn_final", fg_spv_nn_final, sizeof(fg_spv_nn_final) },
		{ "sp_clear", fg_spv_sp_clear, sizeof(fg_spv_sp_clear) },
		{ "sp_depth", fg_spv_sp_depth, sizeof(fg_spv_sp_depth) },
		{ "sp_vec", fg_spv_sp_vec, sizeof(fg_spv_sp_vec) },
		{ "sp_fill", fg_spv_sp_fill, sizeof(fg_spv_sp_fill) },
		{ "sp_gather", fg_spv_sp_gather, sizeof(fg_spv_sp_gather) },
		{ "sp_down", fg_spv_sp_down, sizeof(fg_spv_sp_down) },
		{ "sp_up", fg_spv_sp_up, sizeof(fg_spv_sp_up) },
		{ "sp_final", fg_spv_sp_final, sizeof(fg_spv_sp_final) },
	};
	// the three methods with the common layout, the neural method's six passes and the splat method's eight
	CHECK(sizeof(embedded) / sizeof(embedded[0]) == FG_METHOD_COUNT - 2 + 6 + 8);
	for (const Embedded &e : embedded) {
		CHECK(e.bytes >= 20 && e.bytes % 4 == 0);
		CHECK(e.words[0] == 0x07230203u);
		const std::string path = std::string(shader_dir) + "/" + e.name + ".comp.spv";
		std::ifstream file(path, std::ios::binary);
		const std::vector<char> spv((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		CHECK(!spv.empty());
		CHECK(spv.size() == e.bytes && std::memcmp(spv.data(), e.words, e.bytes) == 0);
	}
}

int main(int argc, char **argv)
{
	if (argc != 2) {
		std::fprintf(stderr, "usage: %s SHADER_DIR\n", argv[0]);
		return 2;
	}
	test_method_names();
	test_create_info();
	test_external_dispatch();
	test_psnr();
	test_embedded_spirv(argv[1]);
	std::printf("framegen host tests: %d checks, %d failed\n", checks, failures);
	return failures ? 1 : 0;
}
