<div align="center">

# PS5 MihawkGen

**Neural frame generation for the PS5, in 432,308 parameters.**
Two colour frames in, the picture between them out. Compute shaders on the PS5's own GPU, no engine data, no motion vectors, no async compute.

[![Parameters](https://img.shields.io/badge/parameters-0.43M-blue)](#how-small-is-it)
[![GPU time](https://img.shields.io/badge/PS5%20GPU%20time-2.2%20ms%20at%204K-brightgreen)](#how-small-is-it)
[![Weights](https://img.shields.io/badge/weights-1.7%20MB-lightgrey)](#how-small-is-it)
[![Licence](https://img.shields.io/badge/licence-GPL--3.0%20%2B%20MIT-orange)](#licence)

</div>

MihawkGen turns a 30 or 60 fps picture into 120 fps on a base PS5: one generated picture between two real frames at 60 fps, three at 30 fps. It works on the *pictures* a title shows, so it needs nothing from the title: no engine data, no motion vectors. It also runs inside RetroArch's Vulkan driver. When a title does hand over depth and motion vectors, a second method (`mvsplat`) uses them.

## Why it is small

| Model | Parameters | Compute for one 288×512 pair | Runs on a PS5? |
| --- | ---: | ---: | --- |
| **MihawkGen network** | **0.43 M** | **4.4 GFLOPs** | **yes: about 2.2 ms at 3840×2160** |
| AMT-S | 3.0 M | 32 GFLOPs | not tested |
| IFRNet | 5.0 M | 57 GFLOPs | not tested |
| IFRNet-L | 19.7 M | 226 GFLOPs | no (about 50× the compute) |
| GIMM-VFI-R | 19.8 M | RAFT, 20 iterations | no |

The network estimates a flow and a mask at four scales, warps both real frames with it, and mixes them. It cannot paint a pixel that is not in one of the two frames; a last shader pass keeps still regions exact.

## Benchmarks

The honest kind: the same triplets (frame, the real picture between, next frame) for every method, never-trained clips only, a hash audit of the benchmark against the training clips, and one scorer for all. PSNR in dB (higher is better), SSIM, LPIPS (lower is better); the average over clip sets.

**33 never-trained 2D clip sets**, the picture at t = ½:

| Method | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| GIMM-VFI-R (20 M, research) | 30.44 | 0.9418 | 0.0420 |
| SGM-VFI (21 M, research) | 30.20 | 0.9372 | 0.0454 |
| **MihawkGen (0.43 M)** | **30.08** | **0.9391** | **0.0428** |
| BiM-VFI (6.9 M, research) | 30.08 | 0.9410 | 0.0426 |
| plain average of the two frames | 29.29 | 0.9223 | 0.0481 |
| FSR 3.1 frame interpolation (no engine data) | 28.82 | 0.9206 | 0.0506 |

**8 never-trained 3D clip sets:**

| Method | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| GIMM-VFI-R | 37.00 | 0.9328 | 0.0675 |
| SGM-VFI | 36.38 | 0.9257 | 0.0621 |
| **MihawkGen** | **35.84** | **0.9178** | **0.0578** |
| BiM-VFI | 35.64 | 0.9295 | 0.0551 |
| plain average | 33.66 | 0.8857 | 0.0716 |
| FSR 3.1 | 31.53 | 0.8840 | 0.0742 |

Three generated pictures between two real ones (t = ¼, ½, ¾) keep the same order: on the 3D set MihawkGen leads the plain average by 1 to 1.6 dB at every instant.

**Reading the tables.** MihawkGen beats FSR 3.1 and the plain average on every metric on both sets, and is within 0.4 dB of a 20-million-parameter research model on 2D clips. The research models are far ahead on 3D (up to 1.2 dB) and cost 7× to 50× the compute. They are measured, not shipped.

### Every situation

`train/situations.py` renders 70 controlled scenes with an exact truth (scrolls, pans, zoom, rotation, parallax, sprites, explosions, particles, fades, HUD over motion, tickers). The pictures are real frames of clips never trained on, moved by code. PSNR in dB:

| Situation | MihawkGen | plain average | FSR 3.1 | IFRNet-L |
| --- | ---: | ---: | ---: | ---: |
| camera pan | 30.0 | 25.1 | 24.6 | 37.6 |
| zoom | 35.6 | 25.8 | 25.3 | 41.3 |
| rotation | 32.5 | 25.9 | 25.7 | 36.3 |
| layered scenes (parallax) | 34.8 | 26.6 | 26.6 | 46.1 |
| sprites | 36.5 | 29.3 | 29.3 | 38.7 |
| explosions | 36.3 | 35.4 | 34.3 | 36.2 |
| HUD over motion | 33.3 | 25.0 | 24.9 | 37.1 |
| fast text tickers | 24.9 | 18.0 | 18.0 | 34.0 |
| side-scrolling | 32.0 | 20.4 | 20.3 | 46.0 |
| fades (small step per frame) | 50.6 | 55.5 | 36.6 | 53.4 |

It is far above the plain average and FSR on motion. The big public model is ahead on large rigid motion, and a fade is better left as the plain blend (the network is 5 dB under it, at 50 dB). A variant with a different last pass closes half of the scroll gap in these scenes and loses on real recordings, so it is not shipped.

### On a base PS5

FrameGen Lab renders a scene at 120 Hz with the exact answer and times every method at 3840×2160:

| Method | PSNR | GPU time |
| --- | ---: | ---: |
| repeat | 27.5 dB | 0.13 ms |
| blend | 32.8 dB | 0.14 ms |
| mvwarp (motion vectors) | 46.1 dB | 0.44 ms |
| **neural (two frames only)** | **46.7 dB** | **2.2 ms** |
| mvsplat (depth and motion vectors) | 50.7 dB | 4.7 ms |

Inside RetroArch the generator held a steady 120 pictures per second with no repeated picture on the 3D content it was tested on.

## Where it fits

```
framegen/              the generator: interface, methods, compute shaders (compiled in), host tests
  models/network.fgw   the trained network (1.7 MB, loaded from a memory blob)
examples/framegenlab/  FrameGen Lab: the scene, the ground truth, the measures
integrations/retroarch RetroArch's Vulkan driver: pacing and groups of generated pictures
train/                 the network: model, trainer, situations suite, benchmark and the public models' adapters
tools/                 the loop, the verdicts, stream analysis
base/ external/ ps5/   the PS5 Vulkan Template foundation (MIT)
```

```bash
framegen/tests/run.sh           # seconds: the device-free tests and the SPIR-V
tools/lab.sh pixels             # the pixel-art motion pass
tools/lab.sh host               # a minute: FrameGen Lab on this PC's Vulkan driver
```

Judge quality on the PC and the PS5; judge speed only on the PS5.

## Training

The network is weights only (432,308 parameters, 1.7 MB): no image or clip is in this repository or in the weights, and the network cannot paint a picture that is not in its input. The trainer in `train/` takes any folder of clips, so you can retrain on material of your own choosing. The network was trained on private clips that are not redistributed, with some of its targets distilled from public interpolation models under permissive licences (RIFE, IFRNet, EMA-VFI, FILM, VFIMamba; credited in `THIRD_PARTY_NOTICES.txt`). The research models used for comparison (GIMM-VFI, BiM-VFI, AMT) are for non-commercial measurement only, are not trained on and are not part of the weights. FSR 3.1 is run as a black box from AMD's FidelityFX SDK, which is not shipped; its licence terms apply to anyone repeating that comparison.

## Licence

`framegen/`, `examples/`, `train/`, `integrations/`: GPL-3.0-or-later. `tools/`, `base/`, `ps5/`: MIT. Third-party libraries under `external/` keep their own licences. See `LICENSE.md` and `THIRD_PARTY_NOTICES.txt`.
