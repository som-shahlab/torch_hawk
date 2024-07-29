#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#include "linear_recurrence_cuda.hh"

namespace {
    #define cudaErrCheck(stat) \
    { cudaErrCheck_((stat), __FILE__, __LINE__); }


void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat),
                file, line);
    }
}

const int WARP_SIZE = 32;

const int LENGTH_PER_WARP = 8;

const int FEATS_PER_WARP = WARP_SIZE * 2;

__global__ void linear_recurrence_forward_kernel_first(const half * __restrict__ a,
                               const half * __restrict__ x, float * __restrict__ s,
                             int n, int k) {
    
    int feature_idx = FEATS_PER_WARP * blockIdx.y + threadIdx.x * 2;

    if (feature_idx >= k) {
        return;
    }

    int length_idx = LENGTH_PER_WARP * threadIdx.y +
                     LENGTH_PER_WARP * blockDim.y * blockIdx.x;

    if (length_idx >= n) {
        return;
    }
    
    a = a + length_idx * k + feature_idx;
    x = x + length_idx * k + feature_idx;
    
    s = s + (length_idx / LENGTH_PER_WARP) * k * 2 + feature_idx;

    float2 current_val = {0, 0};
    float2 current_a = {1, 1};
    
    for (int i = 0; i < LENGTH_PER_WARP; i++) {
        if (length_idx + i >= n) {
            break;
        }
        half2 next_x = *(half2 *)(x + i * k);
        half2 next_a = *(half2 *)(a + i * k);

        float2 next_xf = __half22float2(next_x);
        float2 next_af = __half22float2(next_a);

        current_val.x = current_val.x * next_af.x + next_xf.x;
        current_val.y = current_val.y * next_af.y + next_xf.y;

        current_a.x *= next_af.x;
        current_a.y *= next_af.y;
    }

    s[0] = current_val.x;
    s[1] = current_val.y;

    s[k + 0] = current_a.x;
    s[k + 1] = current_a.y;
}


__global__ void linear_recurrence_forward_kernel_second(
    float * __restrict__ s,
    int n, 
    int k
) {
    
    int feature_idx = FEATS_PER_WARP * blockIdx.y + WARP_SIZE * blockIdx.x + threadIdx.x;

    if (feature_idx >= k) {
        return;
    }

    s = s + feature_idx;

    float current_val = 0;

    int num_segments = (n + LENGTH_PER_WARP - 1) / LENGTH_PER_WARP;

    for (int i = 0; i < num_segments; i++) {
        float next_val = s[i * k * 2];
        float next_a = s[i * k * 2 + k];

        s[i * k * 2] = current_val;

        current_val = current_val * next_a + next_val;
    }
}


__global__ void linear_recurrence_forward_kernel_third(const half * __restrict__ a,
                               const half * __restrict__ x, half* __restrict__ o, const float * __restrict__ s,
                             int n, int k) {
    
    int feature_idx = FEATS_PER_WARP * blockIdx.y + threadIdx.x * 2;

    if (feature_idx >= k) {
        return;
    }

    int length_idx = LENGTH_PER_WARP * threadIdx.y +
                     LENGTH_PER_WARP * blockDim.y * blockIdx.x;

    if (length_idx >= n) {
        return;
    }
    
    a = a + length_idx * k + feature_idx;
    x = x + length_idx * k + feature_idx;
    o = o + length_idx * k + feature_idx;
    
    s = s + (length_idx / LENGTH_PER_WARP) * k * 2 + feature_idx;

    float2 current_val;
    current_val.x = s[0];
    current_val.y = s[1];

    for (int i = 0; i < LENGTH_PER_WARP; i++) {
        if (length_idx + i >= n) {
            break;
        }
        half2 next_x = *(half2 *)(x + i * k);
        half2 next_a = *(half2 *)(a + i * k);

        float2 next_xf = __half22float2(next_x);
        float2 next_af = __half22float2(next_a);

        current_val.x = current_val.x * next_af.x + next_xf.x;
        current_val.y = current_val.y * next_af.y + next_xf.y;

        half2 current_half = __float22half2_rn(current_val);

        *(half2 *)(o + i * k) = current_half;
    }
}



__global__ void linear_recurrence_backward_kernel_first(const half * __restrict__ a,
                               const half * __restrict__ d_o, float * __restrict__ s,
                             int n, int k) {
    
    int feature_idx = FEATS_PER_WARP * blockIdx.y + threadIdx.x * 2;

    if (feature_idx >= k) {
        return;
    }

    int length_idx = LENGTH_PER_WARP * threadIdx.y +
                     LENGTH_PER_WARP * blockDim.y * blockIdx.x;

    if (length_idx >= n) {
        return;
    }
    
    a = a + feature_idx;
    d_o = d_o + feature_idx;
    
    s = s + (length_idx / LENGTH_PER_WARP) * k * 2 + feature_idx;

    float2 current_val = {0, 0};
    float2 current_a = {1, 1};
    
    for (int i = 0; i < LENGTH_PER_WARP; i++) {
        if (length_idx + i >= n) {
            break;
        }
        half2 next_d_o = *(half2 *)(d_o + (n - (length_idx + i) - 1) * k);
        half2 next_a = {0, 0};
        if (length_idx + i != 0) {
            next_a = *(half2 *)(a + (n - (length_idx + i) - 1 + 1) * k);
        }

        float2 next_d_of = __half22float2(next_d_o);
        float2 next_af = __half22float2(next_a);

        current_val.x = current_val.x * next_af.x + next_d_of.x;
        current_val.y = current_val.y * next_af.y + next_d_of.y;

        current_a.x *= next_af.x;
        current_a.y *= next_af.y;
    }

    s[0] = current_val.x;
    s[1] = current_val.y;

    s[k + 0] = current_a.x;
    s[k + 1] = current_a.y;
}

__global__ void linear_recurrence_backward_kernel_third(
    const half * __restrict__ a,
    const half* __restrict__ o, 
    const float * __restrict__ s,

    half * __restrict__ d_a,
    half * __restrict__ d_x, 
    const half* __restrict__ d_o, 

                             int n, int k) {
    int feature_idx = FEATS_PER_WARP * blockIdx.y + threadIdx.x * 2;

    if (feature_idx >= k) {
        return;
    }

    int length_idx = LENGTH_PER_WARP * threadIdx.y +
                     LENGTH_PER_WARP * blockDim.y * blockIdx.x;

    if (length_idx >= n) {
        return;
    }
    
    a = a + feature_idx;
    o = o + feature_idx;
    d_a = d_a + feature_idx;
    d_x = d_x + feature_idx;
    d_o = d_o + feature_idx;
    
    s = s + (length_idx / LENGTH_PER_WARP) * k * 2 + feature_idx;

    float2 current_val;
    current_val.x = s[0];
    current_val.y = s[1];

    for (int i = 0; i < LENGTH_PER_WARP; i++) {
        if (length_idx + i >= n) {
            break;
        }
        half2 next_d_o = *(half2 *)(d_o + (n - (length_idx + i) - 1) * k);
        half2 next_a = {0, 0};
        if (length_idx + i != 0) {
            next_a = *(half2 *)(a + (n - (length_idx + i) - 1 + 1) * k);
        }

        float2 next_d_of = __half22float2(next_d_o);
        float2 next_af = __half22float2(next_a);

        current_val.x = current_val.x * next_af.x + next_d_of.x;
        current_val.y = current_val.y * next_af.y + next_d_of.y;

        half2 current_half = __float22half2_rn(current_val);

        *(half2 *)(d_x + (n - (length_idx + i) - 1) * k) = current_half;


        half2 current_o = {0, 0};
        if (length_idx + i != n - 1) {
            current_o = *(half2 *)(o + (n - (length_idx + i) - 2) * k);    
        }

        *(half2 *)(d_a + (n - (length_idx + i) - 1) * k) = __hmul2(current_o, current_half);
    }
}


}

void linear_recurrence_forward_cuda(
                        void *a,
                        void *x, void *o, void *s,
                        int n, int k) {
    dim3 blockDim, gridDim;
    
    blockDim.x = 32;
    blockDim.y = 8;

    gridDim.x = (n + blockDim.y * LENGTH_PER_WARP - 1) / (blockDim.y * LENGTH_PER_WARP);
    gridDim.y = (k + FEATS_PER_WARP - 1) / FEATS_PER_WARP;

    linear_recurrence_forward_kernel_first<<<gridDim, blockDim>>>((const half*) a, (const half*) x, (float*) s, n, k);

    blockDim.x = 32;
    blockDim.y = 1;

    gridDim.x = 2;
    gridDim.y = (k + FEATS_PER_WARP - 1) / FEATS_PER_WARP;

    linear_recurrence_forward_kernel_second<<<gridDim, blockDim>>>((float*) s, n, k);


    blockDim.x = 32;
    blockDim.y = 8;

    gridDim.x = (n + blockDim.y * LENGTH_PER_WARP - 1) / (blockDim.y * LENGTH_PER_WARP);
    gridDim.y = (k + FEATS_PER_WARP - 1) / FEATS_PER_WARP;

    linear_recurrence_forward_kernel_third<<<gridDim, blockDim>>>((const half*) a, (const half*) x, (half*) o, (const float*) s, n, k);
}

void linear_recurrence_backward_cuda(
        void *a,
        void *o, void *s,
        void* d_a, void* d_x, void* d_o,
        int n, int k) {
    dim3 blockDim, gridDim;
    
    blockDim.x = 32;
    blockDim.y = 8;

    gridDim.x = (n + blockDim.y * LENGTH_PER_WARP - 1) / (blockDim.y * LENGTH_PER_WARP);
    gridDim.y = (k + FEATS_PER_WARP - 1) / FEATS_PER_WARP;

    linear_recurrence_backward_kernel_first<<<gridDim, blockDim>>>((const half*) a, (const half*) d_o, (float*) s, n, k);

    blockDim.x = 32;
    blockDim.y = 1;

    gridDim.x = 2;
    gridDim.y = (k + FEATS_PER_WARP - 1) / FEATS_PER_WARP;

    linear_recurrence_forward_kernel_second<<<gridDim, blockDim>>>((float*) s, n, k);


    blockDim.x = 32;
    blockDim.y = 8;

    gridDim.x = (n + blockDim.y * LENGTH_PER_WARP - 1) / (blockDim.y * LENGTH_PER_WARP);
    gridDim.y = (k + FEATS_PER_WARP - 1) / FEATS_PER_WARP;

    linear_recurrence_backward_kernel_third<<<gridDim, blockDim>>>((const half*) a, (const half*) o, (const float*) s, (half*) d_a, (half*) d_x, (const half*) d_o, n, k);
}

// void conv1d_backward_cuda(
//                         void *x,
//                         void *w, void* s, void *d_o,
//                         void *d_x, void *d_w,
//                         int n, int k) {
//     dim3 blockDim, gridDim;
    
//     blockDim.x = 32;
//     blockDim.y = 8;

//     gridDim.x = (n + 8 * LENGTH_PER_WARP - 1) / (8 * LENGTH_PER_WARP);
//     gridDim.y = (k + FEATS_PER_WARP - 1) / FEATS_PER_WARP;

//     conv1d_backward_kerenl<<<gridDim, blockDim>>>(
//         (const half*) x, (const half*) w, (const uint8_t*) s, 
//         (const half*) d_o, 
//         (half*) d_x, (float*) d_w, 
//         n, k);
// }

std::vector<int> linear_recurrence_scratch_space(int n, int k) {
    int num_segments = (n + LENGTH_PER_WARP - 1) / LENGTH_PER_WARP;
    return {num_segments, 2, k}; 
}