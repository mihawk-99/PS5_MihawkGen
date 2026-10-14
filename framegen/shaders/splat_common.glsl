// PS5_FrameGen - what the passes of method "mvsplat" share (fg_splat.cpp makes the layout): a descriptor set of five
// sampled images and two storage images, and one push-constant block; a pass uses the bindings it needs.
//
// Copyright (C) 2026 Mihawk
// SPDX-License-Identifier: GPL-3.0-or-later

layout(local_size_x = 8, local_size_y = 8) in;

layout(push_constant) uniform Params {
	ivec2 size;   // the pass's own size in pixels (the frames' size, or a pyramid level's)
	float t;      // where the generated frame lies
	int step;     // fill: the distance of the pixels it looks at
	int first;    // fill: 1 for the first pass (the winning-depth image says what is valid)
	int top;      // up: 1 at the coarsest level (nothing coarser to reach)
	float tol;    // how far a frame's depth may differ from the field's for the frame to show the surface
	int inverted; // the depth buffer's 1 is the near plane
	float nearZ;  // the projection's planes (farZ 0: the depth is already a linear view depth, smaller nearer)
	float farZ;
} pc;

const uint kNoDepth = 0xFFFFFFFFu;

// A depth buffer's value as a distance in front of the camera (smaller is nearer), whatever the buffer holds
float view_depth(float d)
{
	if (pc.farZ <= 0.0) {
		return max(d, 0.0);
	}
	const float n = pc.inverted != 0 ? 1.0 - d : d;
	return pc.nearZ * pc.farZ / max(pc.farZ - n * (pc.farZ - pc.nearZ), 1e-6);
}

// Whether two view depths are one surface: within tol of each other (of the farther, when they are distances from a projection)
bool same_surface(float a, float b)
{
	return abs(a - b) < pc.tol * (pc.farZ > 0.0 ? max(a, b) : 1.0);
}
