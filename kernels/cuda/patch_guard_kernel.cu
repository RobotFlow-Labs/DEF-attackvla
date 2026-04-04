/**
 * DEF-attackvla: Fused VLA Defense Kernels
 *
 * 1. fused_smooth_clamp — Randomized smoothing: add Gaussian noise + clamp in one pass
 * 2. patch_guard_tv — Local total variation scoring for patch detection
 * 3. fused_normalize — Dual-normalization (two means/stds) fused in one kernel
 *
 * Target: 2-4x speedup over separate PyTorch ops for real-time defense.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

/**
 * Fused randomized smoothing: noise + clamp in one kernel.
 * Each thread generates its own random noise using cuRAND.
 */
__global__ void fused_smooth_clamp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float sigma,
    float clamp_lo,
    float clamp_hi,
    unsigned long long seed,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    float noise = curand_normal(&state) * sigma;
    float val = input[idx] + noise;
    output[idx] = fminf(fmaxf(val, clamp_lo), clamp_hi);
}

/**
 * Local TV computation: computes |dx| + |dy| for each pixel.
 * Output is a per-pixel TV score map.
 */
__global__ void local_tv_kernel(
    const float* __restrict__ image,
    float* __restrict__ tv_map,
    int C, int H, int W
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float tv = 0.0f;
    for (int c = 0; c < C; c++) {
        float val = image[c * H * W + y * W + x];
        float dx = 0.0f, dy = 0.0f;
        if (x + 1 < W) dx = fabsf(image[c * H * W + y * W + (x + 1)] - val);
        if (y + 1 < H) dy = fabsf(image[c * H * W + (y + 1) * W + x] - val);
        tv += dx + dy;
    }
    tv_map[y * W + x] = tv / (float)C;
}

/**
 * Fused dual normalization: given two (mean, std) pairs, produce
 * a concatenated output [norm0, norm1] along the channel dimension.
 * Input: (C, H, W), Output: (2*C, H, W)
 */
__global__ void fused_dual_normalize_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ mean0,
    const float* __restrict__ std0,
    const float* __restrict__ mean1,
    const float* __restrict__ std1,
    int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H * W;
    if (idx >= total) return;

    int c = idx / (H * W);
    int spatial = idx % (H * W);
    float val = input[idx];

    // First normalization
    output[c * H * W + spatial] = (val - mean0[c]) / std0[c];
    // Second normalization (offset by C channels)
    output[(C + c) * H * W + spatial] = (val - mean1[c]) / std1[c];
}


// --- PyTorch bindings ---

torch::Tensor fused_smooth_clamp(
    torch::Tensor input,
    float sigma,
    float clamp_lo,
    float clamp_hi,
    int64_t seed
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    auto output = torch::empty_like(input);
    int numel = input.numel();
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    fused_smooth_clamp_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        sigma, clamp_lo, clamp_hi,
        (unsigned long long)seed, numel
    );
    return output;
}

torch::Tensor local_tv_map(torch::Tensor image) {
    TORCH_CHECK(image.is_cuda(), "image must be CUDA tensor");
    TORCH_CHECK(image.dim() == 3, "image must be (C, H, W)");
    int C = image.size(0), H = image.size(1), W = image.size(2);

    auto tv_map = torch::zeros({H, W}, image.options());
    dim3 threads(16, 16);
    dim3 blocks((W + 15) / 16, (H + 15) / 16);

    local_tv_kernel<<<blocks, threads>>>(
        image.data_ptr<float>(),
        tv_map.data_ptr<float>(),
        C, H, W
    );
    return tv_map;
}

torch::Tensor fused_dual_normalize(
    torch::Tensor input,
    torch::Tensor mean0, torch::Tensor std0,
    torch::Tensor mean1, torch::Tensor std1
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    int C = input.size(0), H = input.size(1), W = input.size(2);

    auto output = torch::empty({2 * C, H, W}, input.options());
    int total = C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_dual_normalize_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        mean0.data_ptr<float>(), std0.data_ptr<float>(),
        mean1.data_ptr<float>(), std1.data_ptr<float>(),
        C, H, W
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_smooth_clamp", &fused_smooth_clamp,
          "Fused randomized smoothing + clamp (CUDA)");
    m.def("local_tv_map", &local_tv_map,
          "Per-pixel local total variation map (CUDA)");
    m.def("fused_dual_normalize", &fused_dual_normalize,
          "Fused dual normalization for VLA preprocessing (CUDA)");
}
