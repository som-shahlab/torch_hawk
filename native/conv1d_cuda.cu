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

const int LENGTH_PER_WARP = 64;

const int KERNEL_WIDTH = 4;

const int FEATS_PER_WARP = WARP_SIZE * 2;

__global__ void conv1d_forward_kerenl(const __restrict__ half *x,
                               const __restrict__ half *w, const __restrict__ uint8_t *s, __restrict__ half *o,
                             int n, int k) {
    
    printf("Sure1\n");

    int feature_idx = FEATS_PER_WARP * blockIdx.y + threadIdx.x * 2;

    printf("What in the world?\n");

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
            prior_state[++current_offset] = next;
            segments[current_offset] = s[-(KERNEL_WIDTH - i - 1)];
        }
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

        printf("Writing it! %f\n", (float) result.x);

        current_offset = (current_offset + 1) % KERNEL_WIDTH;
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

    cudaErrCheck(cudaDeviceSynchronize());

    printf("Running with wmma... %d %d\n", gridDim.x, gridDim.y);
    conv1d_forward_kerenl<<<gridDim, blockDim>>>((const __restrict__ half*) x, (const __restrict__ half*) w, (const __restrict__ uint8_t*) s, (__restrict__ half*) o, n, k);
   cudaErrCheck(cudaPeekAtLastError());
    cudaErrCheck(cudaDeviceSynchronize());
}