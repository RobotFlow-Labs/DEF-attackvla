#include <metal_stdlib>
using namespace metal;

kernel void patch_guard_score(
    device const float* image [[buffer(0)]],
    device const float* mask [[buffer(1)]],
    device float* out_score [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out_score[gid] = image[gid] * mask[gid];
}
