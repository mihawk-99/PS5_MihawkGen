#version 450
// FrameGen Lab - the model's vertices, and where each was a step before.
//
// Copyright (C) 2026 Mihawk
// SPDX-License-Identifier: GPL-3.0-or-later
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;

layout (location = 0) out vec3 outWorldPos;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec2 outUV;
layout (location = 3) out vec4 outClip;
layout (location = 4) out vec4 outPrevClip;

void main()
{
	vec4 world = scene.model * vec4(inPos, 1.0);
	outWorldPos = world.xyz;
	outNormal = mat3(scene.model) * inNormal;
	outUV = inUV;
	outClip = scene.projection * scene.view * world;
	outPrevClip = scene.prevProjection * scene.prevView * scene.prevModel * vec4(inPos, 1.0);
	gl_Position = outClip;
}
