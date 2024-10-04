#include <cuda_fp16.h>
#include <cuda_bf16.h>
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


__device__ float2 cast_to_float2(half2 v) {
    return __half22float2(v);
}


__device__ float2 cast_to_float2(float2 v) {
    return v;
}

__device__ float2 cast_to_float2(__nv_bfloat162 v) {
    return __bfloat1622float2(v);
}

__device__ float2 mul2(float2 a, float2 b) {
    float2 result = {a.x * b.x, a.y * b.y};
    return result;
}

__device__ half2 mul2(half2 a, half2 b) {
    return __hmul2(a, b);
}


__device__ __nv_bfloat162 mul2(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __hmul2(a, b);
}


__device__ float2 fma2(float2 a, float2 b, float2 c) {
    float2 result = {c.x + a.x * b.x, c.y + a.y * b.y};
    return result;
}

__device__ half2 fma2(half2 a, half2 b, half2 c) {
    return __hfma2(a, b, c);
}


__device__ __nv_bfloat162 fma2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
    return __hfma2(a, b, c);
}



template<typename SingleVal, typename DoubleVal>
__global__ void conv1d_forward_kerenl(const SingleVal * __restrict__ x,
                               const SingleVal * __restrict__ w, 
                               const uint8_t * __restrict__ s, SingleVal * __restrict__ o,
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

    DoubleVal kernel_weights[KERNEL_WIDTH] = {};
    DoubleVal prior_state[KERNEL_WIDTH] = {};
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
            DoubleVal next = *(DoubleVal *)(x - (KERNEL_WIDTH - i - 1) * k);
            prior_state[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = next;
            segments[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = s[-(KERNEL_WIDTH - i - 1)];
        }
        current_offset++;
    }

    for (int i = 0; i < LENGTH_PER_WARP; i++) {
        if (length_idx + i >= n) {
            break;
        }
        DoubleVal next = *(DoubleVal *)(x + i * k);
        prior_state[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = next;

        uint8_t next_segment = s[i];

        segments[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] =
            next_segment;

        DoubleVal result = {0, 0};

        for (int j = 0; j < KERNEL_WIDTH; j++) {
            if (segments[(current_offset + j) % KERNEL_WIDTH] == next_segment) {
                result =
                    fma2(prior_state[(current_offset + j) % KERNEL_WIDTH],
                            kernel_weights[j], result);
            }
        }

        *(DoubleVal *)(o + i * k) = result;

        current_offset = (current_offset + 1) % KERNEL_WIDTH;
    }
}


template<typename SingleVal, typename DoubleVal>
__global__ void conv1d_backward_kerenl(const SingleVal* __restrict__ x,
                               const SingleVal* __restrict__ w, const uint8_t* __restrict__ s, const SingleVal * __restrict__ d_o,
                               SingleVal* __restrict__ d_x, float* __restrict__ d_w,
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

    DoubleVal kernel_weights[KERNEL_WIDTH] = {};
    DoubleVal prior_state[KERNEL_WIDTH] = {};
    DoubleVal prior_x[KERNEL_WIDTH] = {};
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
            DoubleVal next = *(DoubleVal *)(d_o + i * k);
            prior_state[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = next;
            prior_x[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = *(DoubleVal *)(x + i * k);
            segments[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = s[i];
        }
        current_offset++;
    }

    for (int i = 0; i < LENGTH_PER_WARP; i++) {
        if (length_idx + i>= n) {
            break;
        }
        
        if (length_idx + i + KERNEL_WIDTH - 1 < n) {
            prior_state[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = *(DoubleVal *)(d_o + (i + KERNEL_WIDTH - 1) * k);
            prior_x[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = *(DoubleVal *)(x + (i + KERNEL_WIDTH - 1) * k);
            segments[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = s[i + KERNEL_WIDTH  -1];
        } else {
            DoubleVal zero = {0, 0};
            prior_state[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = zero;
            prior_x[(current_offset + KERNEL_WIDTH - 1) % KERNEL_WIDTH] = zero;
        }

        uint8_t current_segment = s[i];

        DoubleVal result = {0, 0};

        for (int j = 0; j < KERNEL_WIDTH; j++) {
            if (segments[(current_offset + j) % KERNEL_WIDTH] == current_segment) {
                result =
                    fma2(prior_state[(current_offset + j) % KERNEL_WIDTH],
                            kernel_weights[j], result);


                float2 val = cast_to_float2(
                    mul2(prior_state[(current_offset + j) % KERNEL_WIDTH], 
                    prior_x[(current_offset) % KERNEL_WIDTH]));
                kernel_derivatives[j].x += val.x;
                kernel_derivatives[j].y += val.y;
            }
        }

        *(DoubleVal *)(d_x + i * k) = result;

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

template<typename SingleVal, typename DoubleVal>
void conv1d_forward_cuda(
                        void *x,
                        void *w, void* s, void *o,
                        int n, int k) {
    dim3 blockDim, gridDim;
    
    blockDim.x = 32;
    blockDim.y = 8;

    gridDim.x = (n + 8 * LENGTH_PER_WARP - 1) / (8 * LENGTH_PER_WARP);
    gridDim.y = (k + FEATS_PER_WARP - 1) / FEATS_PER_WARP;

    conv1d_forward_kerenl<SingleVal, DoubleVal><<<gridDim, blockDim>>>(
                (const SingleVal*) x, (const SingleVal*) w, (const uint8_t* ) s, 
        (SingleVal*) o, n, k);
}

template<typename SingleVal, typename DoubleVal>
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

    conv1d_backward_kerenl<SingleVal, DoubleVal><<<gridDim, blockDim>>>(
        (const SingleVal*) x, (const SingleVal*) w, (const uint8_t*) s, 
        (const SingleVal*) d_o, 
        (SingleVal*) d_x, (float*) d_w, 
        n, k);
}

int conv1d_kernel_size() {
    return KERNEL_WIDTH;
}

void conv1d_forward_cuda_fp16(
                        void *x,
                        void *w, void* s, void *o,
                        int n, int k) {
                            conv1d_forward_cuda<half, half2>(x, w, s, o, n, k);
                        }

void conv1d_backward_cuda_fp16(
                        void *x,
                        void *w, void* s, void *d_o,
                        void *d_x, void *d_w,
                        int n, int k) {
                            conv1d_backward_cuda<half, half2>(x, w, s, d_o, d_x, d_w, n, k);
                        }
                        
void conv1d_forward_cuda_fp32(
                        void *x,
                        void *w, void* s, void *o,
                        int n, int k) {
                            conv1d_forward_cuda<float, float2>(x, w, s, o, n, k);
                        }

void conv1d_backward_cuda_fp32(
                        void *x,
                        void *w, void* s, void *d_o,
                        void *d_x, void *d_w,
                        int n, int k) {
                            conv1d_backward_cuda<float, float2>(x, w, s, d_o, d_x, d_w, n, k);
                        }

void conv1d_forward_cuda_bf16(
                        void *x,
                        void *w, void* s, void *o,
                        int n, int k) {
                            conv1d_forward_cuda<__nv_bfloat16, __nv_bfloat162>(x, w, s, o, n, k);
                        }

void conv1d_backward_cuda_bf16(
                        void *x,
                        void *w, void* s, void *d_o,
                        void *d_x, void *d_w,
                        int n, int k) {
                            conv1d_backward_cuda<__nv_bfloat16, __nv_bfloat162>(x, w, s, d_o, d_x, d_w, n, k);
                        }