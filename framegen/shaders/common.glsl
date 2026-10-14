// PS5_FrameGen - what every generation method's shader binds (fg_vulkan.cpp
// makes this layout). A method uses the bindings it needs.
//
// Copyright (C) 2026 Mihawk
// SPDX-License-Identifier: GPL-3.0-or-later

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0) uniform sampler2D prevColor;
layout(binding = 1) uniform sampler2D nextColor;
layout(binding = 2) uniform sampler2D prevDepth;
layout(binding = 3) uniform sampler2D nextDepth;
layout(binding = 4) uniform sampler2D nextMotion; // backward: next uv minus prev uv
layout(binding = 5, rgba8) uniform writeonly image2D outputImage;

layout(push_constant) uniform Params {
	uvec2 size; // the frames' size in pixels
	float t;    // 0: the previous frame, 1: the next
	uint sourceWidth;  // repeat: the frame's part of prevColor (0: all of it)
	uint sourceHeight;
} params;
