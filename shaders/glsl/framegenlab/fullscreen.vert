#version 450
// FrameGen Lab - one triangle that covers the screen, for full-screen passes.
//
// Copyright (C) 2026 Mihawk
// SPDX-License-Identifier: GPL-3.0-or-later

layout (location = 0) out vec2 outUV;  // 0..1, y down
layout (location = 1) out vec2 outNDC; // -1..1

void main()
{
	outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	outNDC = outUV * 2.0 - 1.0;
	gl_Position = vec4(outNDC, 0.0, 1.0);
}
