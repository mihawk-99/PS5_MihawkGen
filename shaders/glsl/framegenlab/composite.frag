#version 450
// FrameGen Lab - what the screen shows, 2x2: the frame actually rendered
// (top left), the best method's frame (top right: the neural network when it is
// loaded, else mvwarp), the blend baseline (bottom left), and the best method's
// error, four times brighter (bottom right).
//
// Copyright (C) 2026 Mihawk
// SPDX-License-Identifier: GPL-3.0-or-later

layout (set = 0, binding = 0) uniform sampler2D truth;
layout (set = 0, binding = 1) uniform sampler2D generated;
layout (set = 0, binding = 2) uniform sampler2D blended;

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec2 inNDC;

layout (location = 0) out vec4 outColor;

void main()
{
	vec2 quadrant = floor(inUV * 2.0);
	vec2 uv = fract(inUV * 2.0);
	vec3 color;
	if (quadrant.y < 1.0) {
		color = quadrant.x < 1.0 ? texture(truth, uv).rgb : texture(generated, uv).rgb;
	} else if (quadrant.x < 1.0) {
		color = texture(blended, uv).rgb;
	} else {
		color = clamp(abs(texture(generated, uv).rgb - texture(truth, uv).rgb) * 4.0, 0.0, 1.0);
	}
	// A thin dark border between the views
	vec2 border = min(uv, 1.0 - uv) * vec2(textureSize(truth, 0));
	if (min(border.x, border.y) < 2.0) {
		color = vec3(0.04);
	}
	outColor = vec4(color, 1.0);
}
