Data: the Middlebury RubberWhale sequence from OpenCV's test data (ground-truth flow), and for interpolation two real video clips from the same folder, evaluated leave-one-out (frames 0 and 2 in, frame 1 held out as ground truth).

### Flow accuracy (frame10 → frame11)

Average end-point error in pixels (lower is better).

| method | RubberWhale | **mean AEE** | mean AAE° |
|---|---|---|---|
| Flipster LK (numpy) | 0.325 | **0.325** | 10.565 |
| OpenCV Farneback | 0.361 | **0.361** | 12.330 |
| OpenCV DIS (medium) | 0.223 | **0.223** | 7.303 |

### Midpoint interpolation (frame10i11)

PSNR in dB (higher is better).

| method | Corridor-VGA | Street-720p | **mean PSNR** | mean SSIM |
|---|---|---|---|---|
| Cross-dissolve | 28.74 | 20.36 | **24.55** | 0.72 |
| Backward warp (v1 method, fixed) | 34.68 | 21.96 | **28.32** | 0.81 |
| Flipster splat (numpy) | 34.63 | 22.03 | **28.33** | 0.81 |
