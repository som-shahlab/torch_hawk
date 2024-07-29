#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#include "conv1d_cuda.hh"

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

const int KERNEL_WIDTH = 4;

const int FEATS_PER_WARP = WARP_SIZE * 2;

__global__ void conv1d_forward_kerenl(const half * __restrict__ x,
                               const half * __restrict__ w, const uint8_t * __restrict__ s, half * __restrict__ o,
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

    w = w + KERNEL_WIDTH * feature_idx;
    x = x + length_idx * k + feature_idx;
    o = o + length_idx * k + feature_idx;
    s = s + length_idx;

    half2 kernel_weights[KERNEL_WIDTH] = {};
    half2 prior_state[KERNEL_WIDTH] = {};
    uint8_t segments[KERNEL_WIDTH] = {};

    int current_offset = 0;

    for (int i = 0; i < KERNEL_WIDTH; i++) {
        kernel_weights[i].x = *(w + i);
    }

    for (int i = 0; i < KERNEL_WIDTH; i++) {
        kernel_weights[i].y = *(w + i + KERNEL_WIDTH);
    }

    for (int i = 0; i < KERNEL_WIDTH - 1; i++) {
        if (length_idx - (KERNEL_WIDTH - i - 1) >= 0) {
            half2 next = *(half2 *)(x - (KERNEL_WIDTH - i - 1) * k);
            prior_state[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = next;
            segments[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = s[-(KERNEL_WIDTH - i - 1)];
        }
        current_offset++;
    }

    for (int i = 0; i < LENGTH_PER_WARP; i++) {
        if (length_idx + i >= n) {
            break;
        }
        half2 next = *(half2 *)(x + i * k);
        prior_state[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = next;

        uint8_t next_segment = s[i];

        segments[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] =
            next_segment;

        half2 result = __halves2half2(0, 0);

        for (int j = 0; j < KERNEL_WIDTH; j++) {
            if (segments[(current_offset + j) % KERNEL_WIDTH] == next_segment) {
                result =
                    __hfma2(prior_state[(current_offset + j) % KERNEL_WIDTH],
                            kernel_weights[j], result);
            }
        }

        *(half2 *)(o + i * k) = result;

        current_offset = (current_offset + 1) % KERNEL_WIDTH;
    }
}


__global__ void conv1d_backward_kerenl(const half* __restrict__ x,
                               const half* __restrict__ w, const uint8_t* __restrict__ s, const half * __restrict__ d_o,
                               half* __restrict__ d_x, float* __restrict__ d_w,
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

    w = w + KERNEL_WIDTH * feature_idx;
    x = x + length_idx * k + feature_idx;
    s = s + length_idx;
    
    d_o = d_o + length_idx * k + feature_idx;
    d_x = d_x + length_idx * k + feature_idx;
    d_w = d_w + KERNEL_WIDTH * feature_idx;

    half2 kernel_weights[KERNEL_WIDTH] = {};
    half2 prior_state[KERNEL_WIDTH] = {};
    half2 prior_x[KERNEL_WIDTH] = {};
    uint8_t segments[KERNEL_WIDTH] = {};
    float2 kernel_derivatives[KERNEL_WIDTH] = {};

    int current_offset = 0;

    for (int i = 0; i < KERNEL_WIDTH; i++) {
        kernel_weights[KERNEL_WIDTH - i - 1].x = *(w + i);
    }

    for (int i = 0; i < KERNEL_WIDTH; i++) {
        kernel_weights[KERNEL_WIDTH - i - 1].y = *(w + i + KERNEL_WIDTH);
    }

    for (int i = 0; i < KERNEL_WIDTH - 1; i++) {
        if (length_idx + i < n) {
            half2 next = *(half2 *)(d_o + i * k);
            prior_state[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = next;
            prior_x[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = *(half2 *)(x + i * k);
            segments[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = s[i];
        }
        current_offset++;
    }

    for (int i = 0; i < LENGTH_PER_WARP; i++) {
        if (length_idx + i>= n) {
            break;
        }
        
        if (length_idx + i + KERNEL_WIDTH - 1 < n) {
            prior_state[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = *(half2 *)(d_o + (i + KERNEL_WIDTH - 1) * k);
            prior_x[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = *(half2 *)(x + (i + KERNEL_WIDTH - 1) * k);
            segments[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = s[i + KERNEL_WIDTH  -1];
        } else {
            prior_state[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = __halves2half2(0, 0);
            prior_x[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = __halves2half2(0, 0);
        }

        uint8_t current_segment = s[i];

        half2 result = __halves2half2(0, 0);

        for (int j = 0; j < KERNEL_WIDTH; j++) {
            if (segments[(current_offset + j) % KERNEL_WIDTH] == current_segment) {
                result =
                    __hfma2(prior_state[(current_offset + j) % KERNEL_WIDTH],
                            kernel_weights[j], result);


                float2 val = __half22float2(__hmul2(prior_state[(current_offset + j) % KERNEL_WIDTH], prior_x[(current_offset) % KERNEL_WIDTH]));
                kernel_derivatives[j].x += val.x;
                kernel_derivatives[j].y += val.y;
            }
        }

        *(half2 *)(d_x + i * k) = result;

        current_offset = (current_offset + 1) % KERNEL_WIDTH;
    }

    for (int i = 0; i < KERNEL_WIDTH; i++) {
        float2 current = kernel_derivatives[KERNEL_WIDTH -i - 1];
        #define FULL_MASK 0xffffffff
        for (int offset = 16; offset > 0; offset /= 2) {
            current.x += __shfl_down_sync(FULL_MASK, current.x, offset);
            current.y += __shfl_down_sync(FULL_MASK, current.y, offset);
        }

        if (threadIdx.x == 0) {
            // Warp leader
            atomicAdd(d_w + i, current.x);
            atomicAdd(d_w + i + KERNEL_WIDTH, current.y);
        }
    }
}

}

void conv1d_forward_cuda(
                        void *x,
                        void *w, void* s, void *o,
                        int n, int k) {
    dim3 blockDim, gridDim;
    
    blockDim.x = 32;
    blockDim.y = 8;

    gridDim.x = (n + 8 * LENGTH_PER_WARP - 1) / (8 * LENGTH_PER_WARP);
    gridDim.y = (k + FEATS_PER_WARP - 1) / FEATS_PER_WARP;

    conv1d_forward_kerenl<<<gridDim, blockDim>>>(
                (const half*) x, (const half*) w, (const uint8_t* ) s, 
        (half*) o, n, k);
}

void conv1d_backward_cuda(
                        void *x,
                        void *w, void* s, void *d_o,
                        void *d_x, void *d_w,
                        int n, int k) {
    dim3 blockDim, gridDim;
    
    blockDim.x = 32;
    blockDim.y = 8;

    gridDim.x = (n + 8 * LENGTH_PER_WARP - 1) / (8 * LENGTH_PER_WARP);
    gridDim.y = (k + FEATS_PER_WARP - 1) / FEATS_PER_WARP;

    conv1d_backward_kerenl<<<gridDim, blockDim>>>(
        (const half*) x, (const half*) w, (const uint8_t*) s, 
        (const half*) d_o, 
        (half*) d_x, (float*) d_w, 
        n, k);
}

int conv1d_kernel_size() {
    return KERNEL_WIDTH;
}