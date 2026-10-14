#version 450
// FrameGen Lab - the background: a patterned sphere at infinity, so the
// camera's turns move it across the screen. Its motion vector comes from
// projecting the same direction with the previous step's rotation (at
// infinity only rotation moves a point).
//
// Copyright (C) 2026 Mihawk
// SPDX-License-Identifier: GPL-3.0-or-later
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec2 inNDC;

layout (location = 0) out vec4 outColor;
layout (location = 1) out vec2 outMotion;

const float PI = 3.14159265358979;

void main()
{
	vec4 far = scene.skyInverse * vec4(inNDC, 1.0, 1.0);
	vec3 dir = normalize(far.xyz / far.w);
	// Longitude and latitude cells, with thin lines between them; the world's Y
	// is down, so the sky (lighter) is at -Y
	vec2 cell = vec2(atan(dir.z, dir.x), asin(clamp(dir.y, -1.0, 1.0))) * (12.0 / PI);
	float checker = mod(floor(cell.x) + floor(cell.y), 2.0);
	vec2 edge = abs(fract(cell) - 0.5);
	float line = smoothstep(0.47, 0.49, max(edge.x, edge.y));
	vec3 ground = mix(vec3(0.20, 0.17, 0.14), vec3(0.36, 0.31, 0.25), checker);
	vec3 sky = mix(vec3(0.30, 0.38, 0.52), vec3(0.52, 0.60, 0.72), checker);
	vec3 color = mix(ground, sky, smoothstep(0.05, -0.05, dir.y));
	color = mix(color, vec3(0.85, 0.82, 0.70), line * 0.6);
	outColor = vec4(color, 1.0);
	vec4 prev = scene.prevSky * vec4(dir, 0.0);
	outMotion = (inNDC - prev.xy / prev.w) * 0.5;
}
