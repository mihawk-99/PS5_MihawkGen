/*
 * FrameGen Lab - the frame generator's test bench (PS5_FrameGen's harness).
 *
 * The scene (a spinning lantern before a patterned sphere, under a camera that
 * turns, bobs and dollies) is rendered one 1/120 s step at a time into a ring
 * of three offscreen frames: colour, depth and backward motion vectors (each spanning two steps). After
 * every even step k, each generation method (framegen.h) makes frame k-1 from
 * frames k-2 and k at t = 0.5, and is measured against the frame actually
 * rendered at k-1: PSNR over RGB, and GPU time from timestamps. klog gets one
 * line per measured frame and a summary per method at the end; the run's
 * verdict comes from tools/check-framegen.py. The screen shows, 2x2, the frame
 * measured, the blend and repeat baselines, and the blend's error.
 *
 * Synchronisation lives in this file alone (the generator records none): every
 * image's transitions are next to the commands that need them, in record().
 * Results come back a frame in flight later, after that frame's fence.
 *
 * Grown from the PS5 Vulkan Template starter (MIT, Copyright (C) 2026 Mihawk).
 *
 * Copyright (C) 2026 Mihawk
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"

class VulkanExample : public VulkanExampleBase
{
public:
	// The scene's clock: one step is 1/120 s, a display of 120 Hz showing
	// every frame; a 60 Hz game's frames are the even steps
	static constexpr float kStep = 1.0f / 120.0f;
	static constexpr uint32_t kRing = 3;
	static constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
	static constexpr VkFormat kMotionFormat = VK_FORMAT_R16G16_SFLOAT;
	static constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
	static constexpr uint32_t kErrorGroup = 16; // error.comp's local size

	struct Image {
		VkImage image{ VK_NULL_HANDLE };
		VkDeviceMemory memory{ VK_NULL_HANDLE };
		VkImageView view{ VK_NULL_HANDLE };
	};
	struct RingFrame {
		Image color;
		Image depth;
		Image motion;
	};
	std::array<RingFrame, kRing> ring;
	// Each method's output; read by the error pass and the composite
	std::array<Image, FG_METHOD_COUNT> generated;

	vkglTF::Model model;

	struct SceneUniforms {
		glm::mat4 projection;
		glm::mat4 view;
		glm::mat4 model;
		glm::mat4 prevProjection;
		glm::mat4 prevView;
		glm::mat4 prevModel;
		glm::mat4 skyInverse;
		glm::mat4 prevSky;
		glm::vec4 lightDirection;
		glm::vec4 cameraPosition;
	};
	std::array<vks::Buffer, maxConcurrentFrames> sceneBuffers;

	VkSampler sampler{ VK_NULL_HANDLE };
	VkDescriptorSetLayout sceneSetLayout{ VK_NULL_HANDLE };
	VkPipelineLayout scenePipelineLayout{ VK_NULL_HANDLE };
	VkPipeline scenePipeline{ VK_NULL_HANDLE };
	VkPipeline environmentPipeline{ VK_NULL_HANDLE };
	std::array<VkDescriptorSet, maxConcurrentFrames> sceneSets{};

	VkDescriptorSetLayout errorSetLayout{ VK_NULL_HANDLE };
	VkPipelineLayout errorPipelineLayout{ VK_NULL_HANDLE };
	VkPipeline errorPipeline{ VK_NULL_HANDLE };
	std::array<std::array<VkDescriptorSet, FG_METHOD_COUNT>, maxConcurrentFrames> errorSets{};
	// Per frame in flight: FG_METHOD_COUNT x errorGroups partial sums, read by the CPU
	std::array<vks::Buffer, maxConcurrentFrames> errorBuffers;
	uint32_t errorGroupsX{ 0 };
	uint32_t errorGroupsY{ 0 };

	VkDescriptorSetLayout compositeSetLayout{ VK_NULL_HANDLE };
	VkPipelineLayout compositePipelineLayout{ VK_NULL_HANDLE };
	VkPipeline compositePipeline{ VK_NULL_HANDLE };
	std::array<VkDescriptorSet, maxConcurrentFrames> compositeSets{};

	// Two timestamps per method per frame in flight, when the queue has them
	VkQueryPool timestamps{ VK_NULL_HANDLE };
	bool timed{ false };
	double timestampNs{ 0.0 };

	fg_context *framegen{ nullptr };
	std::array<bool, FG_METHOD_COUNT> available{};
	// The neural method's passes, timed one by one: a pool of (passes + 1) timestamps for each frame in flight
	VkQueryPool profilePool{ VK_NULL_HANDLE };
	uint32_t profilePasses{ 0 };
	std::vector<double> profileNs;
	uint32_t profileFrames{ 0 };

	// The step being drawn, and what each frame in flight measured
	uint32_t step{ 0 };
	struct Pending {
		bool valid{ false };
		uint32_t frame{ 0 };
	};
	std::array<Pending, maxConcurrentFrames> pending{};
	struct Stats {
		uint32_t frames{ 0 };
		double psnrSum{ 0.0 };
		double psnrMin{ 0.0 };
		double gpuMsSum{ 0.0 };
		uint32_t gpuFrames{ 0 };
		double lastPsnr{ 0.0 };
	};
	std::array<Stats, FG_METHOD_COUNT> stats{};
	// The ring slot of the last frame measured: what the composite shows as the truth
	bool measuredOnce{ false };
	uint32_t shownSlot{ 0 };

	// Export (the data factory's lab source, docs/plan.md phase 2): with FG_EXPORT_DIR
	// set, the first FG_EXPORT_STEPS steps' colour (RGBA8) and motion vectors (RG16F)
	// are written there as color_<step>.rgba and motion_<step>.f16, raw, at the lab's size
	std::string exportDir;
	uint32_t exportSteps{ 0 };
	std::array<vks::Buffer, maxConcurrentFrames> exportColor;
	std::array<vks::Buffer, maxConcurrentFrames> exportMotion;
	std::array<int64_t, maxConcurrentFrames> exportPending{ -1, -1 };

	VulkanExample() : VulkanExampleBase()
	{
		title = "FrameGen Lab";
		camera.type = Camera::CameraType::lookat;
		// Vulkan 1.3's dynamic rendering, as the starter; the base class's helpers
		// for the swapchain pass transition the stencil too
		apiVersion = VK_API_VERSION_1_3;
		useDynamicRendering = true;
		requiresStencil = true;
		deviceCreatepNextChain = &baseDynamicRenderingFeatures;
	}

	~VulkanExample()
	{
		if (!device) {
			return;
		}
		// The device is idle (the launcher waited): the last frames' results are in
		if (prepared) {
			for (uint32_t slot = 0; slot < maxConcurrentFrames; slot++) {
				collect(slot);
			}
			summarize();
		}
		fg_context_destroy(framegen);
		if (profilePool) {
			vkDestroyQueryPool(device, profilePool, nullptr);
		}
		for (VkPipeline pipeline : { scenePipeline, environmentPipeline, errorPipeline, compositePipeline }) {
			vkDestroyPipeline(device, pipeline, nullptr);
		}
		for (VkPipelineLayout layout : { scenePipelineLayout, errorPipelineLayout, compositePipelineLayout }) {
			vkDestroyPipelineLayout(device, layout, nullptr);
		}
		for (VkDescriptorSetLayout layout : { sceneSetLayout, errorSetLayout, compositeSetLayout }) {
			vkDestroyDescriptorSetLayout(device, layout, nullptr);
		}
		vkDestroySampler(device, sampler, nullptr);
		vkDestroyQueryPool(device, timestamps, nullptr);
		for (auto &buffer : sceneBuffers) {
			buffer.destroy();
		}
		for (auto &buffer : errorBuffers) {
			buffer.destroy();
		}
		for (auto &buffer : exportColor) {
			buffer.destroy();
		}
		for (auto &buffer : exportMotion) {
			buffer.destroy();
		}
		for (RingFrame &frame : ring) {
			destroyImage(frame.color);
			destroyImage(frame.depth);
			destroyImage(frame.motion);
		}
		for (Image &image : generated) {
			destroyImage(image);
		}
	}

	void getEnabledFeatures() override
	{
		if (deviceFeatures.samplerAnisotropy) {
			enabledFeatures.samplerAnisotropy = VK_TRUE;
		}
	}

	// --- Images -----------------------------------------------------------------

	void requireFormat(VkFormat format, VkFormatFeatureFlags features, const char *what)
	{
		VkFormatProperties properties;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
		if ((properties.optimalTilingFeatures & features) != features) {
			vks::tools::exitFatal(std::string("FrameGen Lab needs ") + what, -1);
		}
	}

	Image createImage(VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect)
	{
		Image result;
		const VkImageCreateInfo imageInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = format,
			.extent = { width, height, 1 },
			.mipLevels = 1,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = usage,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};
		VK_CHECK_RESULT(vkCreateImage(device, &imageInfo, nullptr, &result.image));
		VkMemoryRequirements requirements;
		vkGetImageMemoryRequirements(device, result.image, &requirements);
		const VkMemoryAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = requirements.size,
			.memoryTypeIndex = vulkanDevice->getMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
		};
		VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &result.memory));
		VK_CHECK_RESULT(vkBindImageMemory(device, result.image, result.memory, 0));
		const VkImageViewCreateInfo viewInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = result.image,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = format,
			.subresourceRange = { aspect, 0, 1, 0, 1 },
		};
		VK_CHECK_RESULT(vkCreateImageView(device, &viewInfo, nullptr, &result.view));
		return result;
	}

	void destroyImage(Image &image)
	{
		vkDestroyImageView(device, image.view, nullptr);
		vkDestroyImage(device, image.image, nullptr);
		vkFreeMemory(device, image.memory, nullptr);
		image = {};
	}

	static void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect, VkImageLayout from,
		VkImageLayout to, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage,
		VkAccessFlags dstAccess)
	{
		const VkImageMemoryBarrier barrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = srcAccess,
			.dstAccessMask = dstAccess,
			.oldLayout = from,
			.newLayout = to,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = image,
			.subresourceRange = { aspect, 0, 1, 0, 1 },
		};
		vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
	}

	void prepareImages()
	{
		requireFormat(kColorFormat, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
			VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT, "RGBA8 as attachment, sampled and storage image");
		requireFormat(kMotionFormat, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT,
			"RG16F as attachment and sampled image");
		requireFormat(kDepthFormat, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT,
			"D32 as depth attachment and sampled image");
		for (RingFrame &frame : ring) {
			frame.color = createImage(kColorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
			frame.motion = createImage(kMotionFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
			frame.depth = createImage(kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
		}
		for (Image &image : generated) {
			image = createImage(kColorFormat, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
		}
		// The outputs start black and readable: the composite shows them before
		// the first measurement (step 2)
		VkCommandBuffer cmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		const VkClearColorValue black{ { 0.0f, 0.0f, 0.0f, 1.0f } };
		const VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		for (Image &image : generated) {
			imageBarrier(cmd, image.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
			vkCmdClearColorImage(cmd, image.image, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &range);
			imageBarrier(cmd, image.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
		}
		vulkanDevice->flushCommandBuffer(cmd, queue, true);
	}

	// --- The scene ----------------------------------------------------------------

	void loadAssets()
	{
		// Khronos's Lantern (CC0), 1.6 tall and centred (ps5/ASSETS.md)
		const uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices | vkglTF::FileLoadingFlags::FlipY;
		model.loadFromFile(getAssetPath() + "models/ps5/lantern/lantern_starter.gltf", vulkanDevice, queue, glTFLoadingFlags);
	}

	// Where the camera is at a step: it turns 24 degrees a second, nods and
	// moves in and out, so the background slides and the lantern's edges
	// uncover what was behind them
	void placeCamera(uint32_t at)
	{
		const float time = static_cast<float>(at) * kStep;
		camera.setPosition(glm::vec3(0.0f, 0.0f, -(3.2f + 0.35f * std::sin(time * 0.9f))));
		camera.setRotation(glm::vec3(-12.0f + 4.0f * std::sin(time * 1.7f), 30.0f + 24.0f * time, 0.0f));
	}

	static glm::mat4 modelAt(uint32_t at)
	{
		return glm::rotate(glm::mat4(1.0f), 1.4f * static_cast<float>(at) * kStep, glm::vec3(0.0f, 1.0f, 0.0f));
	}

	void updateScene(uint32_t at)
	{
		SceneUniforms uniforms{};
		// The previous rendered frame's matrices first: two steps back, as a 60 Hz
		// game's vectors span the whole interval between its frames (steps 0, 1: no
		// motion). The odd steps' vectors are not read; they are the truth.
		const uint32_t before = at >= 2 ? at - 2 : 0;
		placeCamera(before);
		uniforms.prevProjection = camera.matrices.perspective;
		uniforms.prevView = camera.matrices.view;
		uniforms.prevModel = modelAt(before);
		placeCamera(at);
		uniforms.projection = camera.matrices.perspective;
		uniforms.view = camera.matrices.view;
		uniforms.model = modelAt(at);
		// Directions at infinity move only with the view's rotation
		uniforms.skyInverse = glm::inverse(uniforms.projection * glm::mat4(glm::mat3(uniforms.view)));
		uniforms.prevSky = uniforms.prevProjection * glm::mat4(glm::mat3(uniforms.prevView));
		uniforms.lightDirection = glm::vec4(glm::normalize(glm::vec3(-0.45f, 0.75f, -0.5f)), 0.0f);
		uniforms.cameraPosition = glm::inverse(uniforms.view)[3];
		memcpy(sceneBuffers[currentBuffer].mapped, &uniforms, sizeof(uniforms));
	}

	// --- Pipelines and descriptors -----------------------------------------------

	void prepareBuffers()
	{
		for (auto &buffer : sceneBuffers) {
			VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer, sizeof(SceneUniforms)));
			VK_CHECK_RESULT(buffer.map());
		}
		errorGroupsX = (width + kErrorGroup - 1) / kErrorGroup;
		errorGroupsY = (height + kErrorGroup - 1) / kErrorGroup;
		const VkDeviceSize errorBytes = sizeof(float) * FG_METHOD_COUNT * errorGroupsX * errorGroupsY;
		for (auto &buffer : errorBuffers) {
			VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer, errorBytes));
			VK_CHECK_RESULT(buffer.map());
		}
		if (const char *dir = getenv("FG_EXPORT_DIR")) {
			exportDir = dir;
			const char *steps = getenv("FG_EXPORT_STEPS");
			exportSteps = steps ? static_cast<uint32_t>(atoi(steps)) : 0;
			const VkDeviceSize bytes = VkDeviceSize(width) * height * 4;
			for (auto &buffer : exportColor) {
				VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer, bytes));
				VK_CHECK_RESULT(buffer.map());
			}
			for (auto &buffer : exportMotion) {
				VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer, bytes));
				VK_CHECK_RESULT(buffer.map());
			}
			say("framegen: exporting %u steps to %s", exportSteps, exportDir.c_str());
		}
		// GPU time: only when the graphics queue's timestamps count
		const uint32_t validBits = vulkanDevice->queueFamilyProperties[vulkanDevice->queueFamilyIndices.graphics].timestampValidBits;
		timed = validBits > 0 && vulkanDevice->properties.limits.timestampComputeAndGraphics;
		timestampNs = vulkanDevice->properties.limits.timestampPeriod;
		if (timed) {
			const VkQueryPoolCreateInfo queryInfo{
				.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
				.queryType = VK_QUERY_TYPE_TIMESTAMP,
				.queryCount = maxConcurrentFrames * FG_METHOD_COUNT * 2,
			};
			VK_CHECK_RESULT(vkCreateQueryPool(device, &queryInfo, nullptr, &timestamps));
		}
	}

	void setupDescriptors()
	{
		const uint32_t frames = maxConcurrentFrames;
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, frames),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, frames * (2 * FG_METHOD_COUNT + 3)),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, frames * FG_METHOD_COUNT),
		};
		VkDescriptorPoolCreateInfo poolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, frames * (2 + FG_METHOD_COUNT));
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));

		// The scene: set 0 its uniforms; set 1 the model's texture (vkglTF's layout)
		std::vector<VkDescriptorSetLayoutBinding> bindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0),
		};
		VkDescriptorSetLayoutCreateInfo layoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(bindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &sceneSetLayout));
		// The error pass: the truth, the frame tested, the partial sums
		bindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
		};
		layoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(bindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &errorSetLayout));
		// The composite: the truth, the blend, the repeat
		bindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),
		};
		layoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(bindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &compositeSetLayout));

		for (uint32_t i = 0; i < frames; i++) {
			VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &sceneSetLayout, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &sceneSets[i]));
			VkWriteDescriptorSet write = vks::initializers::writeDescriptorSet(sceneSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &sceneBuffers[i].descriptor);
			vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
			for (uint32_t m = 0; m < FG_METHOD_COUNT; m++) {
				allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &errorSetLayout, 1);
				VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &errorSets[i][m]));
			}
			allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &compositeSetLayout, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compositeSets[i]));
		}

		VkSamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.maxLod = 0.0f;
		VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &sampler));
	}

	void preparePipelines()
	{
		// The scene and its background: colour and motion, with depth
		const std::array<VkDescriptorSetLayout, 2> sceneLayouts = { sceneSetLayout, vkglTF::descriptorSetLayoutImage };
		VkPipelineLayoutCreateInfo layoutInfo = vks::initializers::pipelineLayoutCreateInfo(sceneLayouts.data(), 2);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &scenePipelineLayout));

		VkPipelineInputAssemblyStateCreateInfo inputAssembly = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterization = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		const std::array<VkPipelineColorBlendAttachmentState, 2> blendAttachments = {
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
		};
		VkPipelineColorBlendStateCreateInfo colorBlend = vks::initializers::pipelineColorBlendStateCreateInfo(2, blendAttachments.data());
		VkPipelineDepthStencilStateCreateInfo depthStencil = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisample = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStates);
		const std::array<VkFormat, 2> offscreenFormats = { kColorFormat, kMotionFormat };
		VkPipelineRenderingCreateInfo offscreenRendering{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
			.colorAttachmentCount = 2,
			.pColorAttachmentFormats = offscreenFormats.data(),
			.depthAttachmentFormat = kDepthFormat,
		};
		std::array<VkPipelineShaderStageCreateInfo, 2> stages = {
			loadShader(getShadersPath() + "framegenlab/scene.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getShadersPath() + "framegenlab/scene.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
		};
		VkGraphicsPipelineCreateInfo pipelineInfo{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.pNext = &offscreenRendering,
			.stageCount = 2,
			.pStages = stages.data(),
			.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Normal, vkglTF::VertexComponent::UV }),
			.pInputAssemblyState = &inputAssembly,
			.pViewportState = &viewportState,
			.pRasterizationState = &rasterization,
			.pMultisampleState = &multisample,
			.pDepthStencilState = &depthStencil,
			.pColorBlendState = &colorBlend,
			.pDynamicState = &dynamicState,
			.layout = scenePipelineLayout,
		};
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &scenePipeline));

		// The background: one full-screen triangle, drawn first, writing no depth
		VkPipelineVertexInputStateCreateInfo noVertices = vks::initializers::pipelineVertexInputStateCreateInfo();
		depthStencil = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_ALWAYS);
		stages = {
			loadShader(getShadersPath() + "framegenlab/fullscreen.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getShadersPath() + "framegenlab/environment.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
		};
		pipelineInfo.pVertexInputState = &noVertices;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &environmentPipeline));

		// The composite: into the swapchain image, through the base class's pass
		layoutInfo = vks::initializers::pipelineLayoutCreateInfo(&compositeSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &compositePipelineLayout));
		VkPipelineRenderingCreateInfo swapchainRendering{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &swapChain.colorFormat,
			.depthAttachmentFormat = depthFormat,
			.stencilAttachmentFormat = depthFormat,
		};
		colorBlend = vks::initializers::pipelineColorBlendStateCreateInfo(1, blendAttachments.data());
		stages[1] = loadShader(getShadersPath() + "framegenlab/composite.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		pipelineInfo.pNext = &swapchainRendering;
		pipelineInfo.layout = compositePipelineLayout;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &compositePipeline));

		// The error pass
		const VkPushConstantRange errorPush{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * sizeof(uint32_t) };
		layoutInfo = vks::initializers::pipelineLayoutCreateInfo(&errorSetLayout, 1);
		layoutInfo.pushConstantRangeCount = 1;
		layoutInfo.pPushConstantRanges = &errorPush;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &errorPipelineLayout));
		VkComputePipelineCreateInfo computeInfo = vks::initializers::computePipelineCreateInfo(errorPipelineLayout, 0);
		computeInfo.stage = loadShader(getShadersPath() + "framegenlab/error.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
		VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computeInfo, nullptr, &errorPipeline));
	}

	// The trained network: FG_NETWORK (the PC build), else the title's assets, when it has one
	void loadNetwork()
	{
		std::string path = getenv("FG_NETWORK") ? getenv("FG_NETWORK") : getAssetPath() + "framegen/network.fgw";
		std::ifstream file(path, std::ios::binary);
		if (!file) {
			say("framegen: no network at %s: the neural method is not measured", path.c_str());
			return;
		}
		const std::vector<char> blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		char error[256] = "";
		if (fg_context_load_network(framegen, blob.data(), blob.size(), error, sizeof(error)) != VK_SUCCESS) {
			vks::tools::exitFatal(error, -1);
		}
		say("framegen: network %s, %zu bytes", path.c_str(), blob.size());
	}

	void prepareGenerator()
	{
		const fg_create_info info{
			.device = device,
			.physical_device = physicalDevice,
			.width = width,
			.height = height,
			.slots = maxConcurrentFrames,
			.output_format = kColorFormat,
			.get_device_proc_addr = vkGetDeviceProcAddr,
			.get_memory_properties = vkGetPhysicalDeviceMemoryProperties,
		};
		char error[256] = "";
		if (fg_context_create(&info, &framegen, error, sizeof(error)) != VK_SUCCESS) {
			vks::tools::exitFatal(error, -1);
		}
		loadNetwork();
		if (fg_context_enable_mvsplat(framegen, error, sizeof(error)) != VK_SUCCESS) {
			vks::tools::exitFatal(error, -1);
		}
		for (uint32_t m = 0; m < FG_METHOD_COUNT; m++) {
			available[m] = fg_method_available(framegen, fg_method(m));
		}
		profilePasses = available[FG_METHOD_NEURAL] ? fg_neural_pass_count(framegen) : 0;
		if (profilePasses && timed) {
			profileNs.assign(profilePasses, 0.0);
			const VkQueryPoolCreateInfo queryInfo{
				.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
				.queryType = VK_QUERY_TYPE_TIMESTAMP,
				.queryCount = maxConcurrentFrames * (profilePasses + 1),
			};
			VK_CHECK_RESULT(vkCreateQueryPool(device, &queryInfo, nullptr, &profilePool));
		}
	}

	void prepare() override
	{
		VulkanExampleBase::prepare();
		camera.setPerspective(50.0f, (float)width / (float)height, 0.05f, 64.0f);
		loadAssets();
		prepareImages();
		prepareBuffers();
		setupDescriptors();
		preparePipelines();
		prepareGenerator();
		say("framegen: lab %ux%u, %u methods, gpu timing %s", width, height, (unsigned)FG_METHOD_COUNT, timed ? "on" : "off");
		prepared = true;
	}

	// --- Measurements ----------------------------------------------------------

	/* What frame-in-flight `slot` measured, once its fence has passed: one klog
	 * line for the frame */
	void writeExport(uint32_t slot)
	{
		if (exportPending[slot] < 0) {
			return;
		}
		const size_t bytes = size_t(width) * height * 4;
		const struct { const char *name; const char *extension; const vks::Buffer *buffer; } files[] = {
			{ "color", "rgba", &exportColor[slot] }, { "motion", "f16", &exportMotion[slot] } };
		for (const auto &file : files) {
			char path[512];
			snprintf(path, sizeof(path), "%s/%s_%05d.%s", exportDir.c_str(), file.name, static_cast<int>(exportPending[slot]), file.extension);
			if (FILE *f = fopen(path, "wb")) {
				fwrite(file.buffer->mapped, 1, bytes, f);
				fclose(f);
			}
		}
		exportPending[slot] = -1;
	}

	void collect(uint32_t slot)
	{
		writeExport(slot);
		Pending &p = pending[slot];
		if (p.valid && profilePool) {
			std::vector<uint64_t> ticks(profilePasses + 1);
			if (vkGetQueryPoolResults(device, profilePool, slot * (profilePasses + 1), profilePasses + 1, ticks.size() * sizeof(uint64_t),
					ticks.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
				for (uint32_t i = 0; i < profilePasses; i++) {
					profileNs[i] += double(ticks[i + 1] - ticks[i]) * timestampNs;
				}
				profileFrames++;
				if (profileFrames == 60) {
					reportPasses();
				}
			}
		}
		if (!p.valid) {
			return;
		}
		p.valid = false;
		const uint32_t groups = errorGroupsX * errorGroupsY;
		const float *partial = static_cast<const float *>(errorBuffers[slot].mapped);
		uint64_t ticks[FG_METHOD_COUNT * 2]{};
		const bool haveTimes = timed && vkGetQueryPoolResults(device, timestamps, slot * FG_METHOD_COUNT * 2,
			FG_METHOD_COUNT * 2, sizeof(ticks), ticks, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;
		char line[512];
		int used = snprintf(line, sizeof(line), "framegen: frame %u", p.frame);
		for (uint32_t m = 0; m < FG_METHOD_COUNT; m++) {
			if (!available[m]) {
				continue;
			}
			double sse = 0.0;
			for (uint32_t g = 0; g < groups; g++) {
				sse += partial[m * groups + g];
			}
			const double psnr = fg_psnr(sse, uint64_t(width) * height * 3);
			const double ms = haveTimes ? double(ticks[2 * m + 1] - ticks[2 * m]) * timestampNs / 1e6 : -1.0;
			Stats &s = stats[m];
			s.psnrMin = s.frames ? std::min(s.psnrMin, psnr) : psnr;
			s.psnrSum += psnr;
			s.lastPsnr = psnr;
			s.frames++;
			if (ms >= 0.0) {
				s.gpuMsSum += ms;
				s.gpuFrames++;
			}
			used += snprintf(line + used, sizeof(line) - used, " %s %.2f dB %.3f ms", fg_method_name(fg_method(m)), psnr, ms);
		}
		say("%s", line);
	}

	// The per-pass times, from the frames so far (printed in the run's middle: a burst of lines at
	// the very end can be lost with the title's exit)
	void reportPasses()
	{
		for (uint32_t i = 0; i < profilePasses && profileFrames; i++) {
			say("framegen: neural pass %u %s %.3f ms", i, fg_neural_pass_label(framegen, i), profileNs[i] / profileFrames / 1e6);
		}
	}

	void summarize()
	{
		for (uint32_t m = 0; m < FG_METHOD_COUNT; m++) {
			if (!available[m]) {
				continue;
			}
			const Stats &s = stats[m];
			say("framegen: summary %s frames %u psnr_mean %.2f psnr_min %.2f gpu_ms_mean %.3f", fg_method_name(fg_method(m)),
				s.frames, s.frames ? s.psnrSum / s.frames : 0.0, s.psnrMin, s.gpuFrames ? s.gpuMsSum / s.gpuFrames : -1.0);
		}
	}

	// --- A step ---------------------------------------------------------------

	void renderScene(VkCommandBuffer cmd, const RingFrame &frame)
	{
		// The slot's old frame (three steps back) was last read by the generator,
		// the error pass or the composite: discard it once they are done
		const VkPipelineStageFlags readers = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		for (VkImage image : { frame.color.image, frame.motion.image }) {
			imageBarrier(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				readers, 0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
		}
		imageBarrier(cmd, frame.depth.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			readers, 0, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

		std::array<VkRenderingAttachmentInfo, 2> colors{};
		const std::array<VkImageView, 2> colorViews = { frame.color.view, frame.motion.view };
		for (uint32_t i = 0; i < 2; i++) {
			colors[i] = {
				.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
				.imageView = colorViews[i],
				.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.clearValue = { .color = { { 0.0f, 0.0f, 0.0f, 1.0f } } },
			};
		}
		const VkRenderingAttachmentInfo depth{
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = frame.depth.view,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = { .depthStencil = { 1.0f, 0 } },
		};
		const VkRenderingInfo renderingInfo{
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
			.renderArea = { { 0, 0 }, { width, height } },
			.layerCount = 1,
			.colorAttachmentCount = 2,
			.pColorAttachments = colors.data(),
			.pDepthAttachment = &depth,
		};
		vkCmdBeginRendering(cmd, &renderingInfo);
		VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
		vkCmdSetViewport(cmd, 0, 1, &viewport);
		VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
		vkCmdSetScissor(cmd, 0, 1, &scissor);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipelineLayout, 0, 1, &sceneSets[currentBuffer], 0, nullptr);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, environmentPipeline);
		vkCmdDraw(cmd, 3, 1, 0, 0);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipeline);
		model.draw(cmd, vkglTF::RenderFlags::BindImages, scenePipelineLayout);
		vkCmdEndRendering(cmd);

		// The frame is done: sampled from now on, by compute and fragment shaders
		const VkPipelineStageFlags samplers = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		for (VkImage image : { frame.color.image, frame.motion.image }) {
			imageBarrier(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, samplers, VK_ACCESS_SHADER_READ_BIT);
		}
		imageBarrier(cmd, frame.depth.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, samplers, VK_ACCESS_SHADER_READ_BIT);
	}

	/* Every method makes frame `truth` from its neighbours; each result is
	 * compared with the frame actually rendered, and timed */
	void measure(VkCommandBuffer cmd, const RingFrame &prev, const RingFrame &truth, const RingFrame &next, uint32_t frame)
	{
		for (const Image &image : generated) {
			imageBarrier(cmd, image.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
		}
		const uint32_t firstQuery = currentBuffer * FG_METHOD_COUNT * 2;
		if (timed) {
			vkCmdResetQueryPool(cmd, timestamps, firstQuery, FG_METHOD_COUNT * 2);
		}
		const fg_frame_pair pair{
			.prev_color = prev.color.view,
			.next_color = next.color.view,
			.prev_depth = prev.depth.view,
			.next_depth = next.depth.view,
			.next_motion = next.motion.view,
			.t = 0.5f,
			.depth_near = 0.05f,   // (the lab camera: setPerspective(50, aspect, 0.05, 64))
			.depth_far = 64.0f,
		};
		for (uint32_t m = 0; m < FG_METHOD_COUNT; m++) {
			if (m == FG_METHOD_NEURAL && profilePool) {
				const uint32_t base = currentBuffer * (profilePasses + 1);
				vkCmdResetQueryPool(cmd, profilePool, base, profilePasses + 1);
				fg_context_set_profile(framegen, profilePool, base);
			}
			// Bottom of pipe on both sides: the time between the work before it
			// finishing and the method's own work finishing
			if (timed) {
				vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, firstQuery + 2 * m);
			}
			fg_record_generate(framegen, cmd, currentBuffer, fg_method(m), &pair, generated[m].view);
			if (timed) {
				vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, firstQuery + 2 * m + 1);
			}
		}
		for (const Image &image : generated) {
			imageBarrier(cmd, image.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
		}

		const uint32_t groups = errorGroupsX * errorGroupsY;
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, errorPipeline);
		for (uint32_t m = 0; m < FG_METHOD_COUNT; m++) {
			const VkDescriptorSet set = errorSets[currentBuffer][m];
			VkDescriptorImageInfo truthInfo{ sampler, truth.color.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
			VkDescriptorImageInfo testedInfo{ sampler, generated[m].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
			const std::array<VkWriteDescriptorSet, 3> writes = {
				vks::initializers::writeDescriptorSet(set, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &truthInfo),
				vks::initializers::writeDescriptorSet(set, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &testedInfo),
				vks::initializers::writeDescriptorSet(set, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &errorBuffers[currentBuffer].descriptor),
			};
			vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
			const uint32_t push[3] = { width, height, m * groups };
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, errorPipelineLayout, 0, 1, &set, 0, nullptr);
			vkCmdPushConstants(cmd, errorPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
			vkCmdDispatch(cmd, errorGroupsX, errorGroupsY, 1);
		}
		// The sums go to the CPU, which reads them after this frame's fence
		const VkBufferMemoryBarrier toHost{
			.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.buffer = errorBuffers[currentBuffer].buffer,
			.offset = 0,
			.size = VK_WHOLE_SIZE,
		};
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &toHost, 0, nullptr);
		pending[currentBuffer] = { true, frame };
	}

	void composite(VkCommandBuffer cmd, VkImageView truthView)
	{
		const VkDescriptorSet set = compositeSets[currentBuffer];
		std::array<VkDescriptorImageInfo, 3> infos = { {
			{ sampler, truthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
			{ sampler, generated[available[FG_METHOD_NEURAL] ? FG_METHOD_NEURAL : FG_METHOD_MVWARP].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
			{ sampler, generated[FG_METHOD_BLEND].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
		} };
		std::array<VkWriteDescriptorSet, 3> writes;
		for (uint32_t i = 0; i < 3; i++) {
			writes[i] = vks::initializers::writeDescriptorSet(set, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, i, &infos[i]);
		}
		vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
		// The base class's pass clears the swapchain image and the depth buffer
		beginDynamicRendering(cmd);
		VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
		vkCmdSetViewport(cmd, 0, 1, &viewport);
		VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
		vkCmdSetScissor(cmd, 0, 1, &scissor);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipelineLayout, 0, 1, &set, 0, nullptr);
		vkCmdDraw(cmd, 3, 1, 0, 0);
		drawUI(cmd);
		endDynamicRendering(cmd);
	}

	// Copies the step's colour and motion to the host, to be written when this
	// frame in flight's fence has passed (writeExport)
	void exportStep(VkCommandBuffer cmd, const RingFrame &frame)
	{
		const VkPipelineStageFlags samplers = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		const VkBufferImageCopy region{
			.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
			.imageExtent = { width, height, 1 },
		};
		const std::array<std::pair<VkImage, VkBuffer>, 2> copies = { { { frame.color.image, exportColor[currentBuffer].buffer },
			{ frame.motion.image, exportMotion[currentBuffer].buffer } } };
		for (const auto &copy : copies) {
			imageBarrier(cmd, copy.first, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
			vkCmdCopyImageToBuffer(cmd, copy.first, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, copy.second, 1, &region);
			imageBarrier(cmd, copy.first, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT, samplers, VK_ACCESS_SHADER_READ_BIT);
		}
		const std::array<VkBuffer, 2> buffers = { exportColor[currentBuffer].buffer, exportMotion[currentBuffer].buffer };
		for (VkBuffer buffer : buffers) {
			const VkBufferMemoryBarrier toHost{
				.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
				.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
				.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.buffer = buffer,
				.size = VK_WHOLE_SIZE,
			};
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &toHost, 0, nullptr);
		}
		exportPending[currentBuffer] = step;
	}

	void record()
	{
		VkCommandBuffer cmd = drawCmdBuffers[currentBuffer];
		VkCommandBufferBeginInfo beginInfo = vks::initializers::commandBufferBeginInfo();
		VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &beginInfo));
		const uint32_t slot = step % kRing;
		renderScene(cmd, ring[slot]);
		if (!exportDir.empty() && step < exportSteps) {
			exportStep(cmd, ring[slot]);
		}
		// After an even step k: frame k-1 can be made from k-2 and k
		if (step >= 2 && step % 2 == 0) {
			const uint32_t truthSlot = (step - 1) % kRing;
			measure(cmd, ring[(step - 2) % kRing], ring[truthSlot], ring[slot], step - 1);
			measuredOnce = true;
			shownSlot = truthSlot;
		}
		composite(cmd, measuredOnce ? ring[shownSlot].color.view : ring[slot].color.view);
		VK_CHECK_RESULT(vkEndCommandBuffer(cmd));
	}

	void render() override
	{
		if (!prepared) {
			return;
		}
		VulkanExampleBase::prepareFrame();
		// This frame in flight's fence has passed: what it measured last time is in
		collect(currentBuffer);
		updateScene(step);
		record();
		VulkanExampleBase::submitFrame();
		step++;
	}

	void OnUpdateUIOverlay(vks::UIOverlay *overlay) override
	{
		if (overlay->header("FrameGen Lab")) {
			overlay->text("Step %u, %.3f s", step, step * kStep);
			for (uint32_t m = 0; m < FG_METHOD_COUNT; m++) {
				const Stats &s = stats[m];
				overlay->text("%s: %.2f dB (mean %.2f)", fg_method_name(fg_method(m)), s.lastPsnr,
					s.frames ? s.psnrSum / s.frames : 0.0);
			}
			if (overlay->button("Quit")) {
				quit = true;
			}
		}
	}
};

VULKAN_EXAMPLE_MAIN()
