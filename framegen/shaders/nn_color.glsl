// PS5_FrameGen - optional HDR10 analysis domain. The output stays in PQ/BT.2020.
// Copyright (C) 2026 Mihawk
// SPDX-License-Identifier: GPL-3.0-or-later
layout(constant_id = 2) const uint HDR10 = 0;

// ST 2084 EOTF, in cd/m2. Clamp the code domain before the fractional powers.
vec3 pqNits(vec3 code)
{
	const vec3 p = pow(clamp(code, 0.0, 1.0), vec3(1.0 / 78.84375));
	return 10000.0 * pow(max(p - 0.8359375, 0.0) / max(18.8515625 - 18.6875 * p, 1e-6), vec3(1.0 / 0.1593017578125));
}

// A fixed 203-nit diffuse-white shoulder, followed by a perceptual encoding.
// This is only for the SDR-trained flow network and still/HUD decisions. It does
// not tone-map the generated picture. Fixed exposure avoids frame-to-frame pumping.
vec3 analysisColor(vec3 code)
{
	if (HDR10 == 0) return code;
	const vec3 linear = pqNits(code);
	const float y = dot(linear, vec3(0.2627, 0.6780, 0.0593));
	return pow(clamp(linear / (203.0 + y), 0.0, 1.0), vec3(1.0 / 2.2));
}

float analysisLuma(vec3 code)
{
	if (HDR10 == 0) return dot(code, vec3(0.299, 0.587, 0.114));
	const float y = dot(pqNits(code), vec3(0.2627, 0.6780, 0.0593));
	return pow(y / (203.0 + y), 1.0 / 2.2);
}
