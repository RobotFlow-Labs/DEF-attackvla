// DEF-attackvla CUDA kernel scaffold.
// Purpose: fuse image clipping + trigger saliency score accumulation.

extern "C" __global__ void patch_guard_score(
    const float* image,
    const float* mask,
    float* out_score,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float value = image[idx] * mask[idx];
    // Placeholder reduction strategy: per-index write.
    out_score[idx] = value;
}
