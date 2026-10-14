#version 450
// FrameGen Lab - the starter's lighting (a warm key light, a cool fill, a soft
// highlight, in linear space, stored as sRGB values in UNORM), and the
// backward motion vector: this pixel's uv minus its uv two steps before.
//
// Copyright (C) 2026 Mihawk
// SPDX-License-Identifier: GPL-3.0-or-later
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"

layout (set = 1, binding = 0) uniform sampler2D baseColor;

layout (location = 0) in vec3 inWorldPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec4 inClip;
layout (location = 4) in vec4 inPrevClip;

layout (location = 0) out vec4 outColor;
layout (location = 1) out vec2 outMotion;

void main()
{
	vec3 n = normalize(inNormal);
	vec3 v = normalize(scene.cameraPosition.xyz - inWorldPos);
	if (dot(n, v) < 0.0) {
		n = -n;
	}
	vec3 l = normalize(-scene.lightDirection.xyz);
	vec3 albedo = pow(texture(baseColor, inUV).rgb, vec3(2.2));
	float key = max(dot(n, l), 0.0);
	float fill = max(dot(n, vec3(0.0, 1.0, 0.0)), 0.0) * 0.25 + 0.15;
	float highlight = pow(max(dot(n, normalize(l + v)), 0.0), 48.0) * 0.35 * key;
	vec3 linearColor = albedo * (key * vec3(1.0, 0.92, 0.80) + fill * vec3(0.55, 0.65, 0.85)) + highlight;
	outColor = vec4(pow(clamp(linearColor, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
	// NDC to uv halves the distance; y is down in both
	outMotion = (inClip.xy / inClip.w - inPrevClip.xy / inPrevClip.w) * 0.5;
}
