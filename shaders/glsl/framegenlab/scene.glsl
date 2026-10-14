// FrameGen Lab - the scene's uniforms, shared by its shaders. Each step's
// matrices and the previous step's, for backward motion vectors.
//
// Copyright (C) 2026 Mihawk
// SPDX-License-Identifier: GPL-3.0-or-later

layout (set = 0, binding = 0) uniform Scene
{
	mat4 projection;
	mat4 view;
	mat4 model;
	mat4 prevProjection;
	mat4 prevView;
	mat4 prevModel;
	mat4 skyInverse;   // inverse(projection * the view's rotation): screen to direction
	mat4 prevSky;      // prevProjection * the previous view's rotation: direction to screen
	vec4 lightDirection;
	vec4 cameraPosition;
} scene;
