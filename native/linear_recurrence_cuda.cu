#include <cuda_fp16.h>
#include <cuda_bf16.h>
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

__device__ float2 cast_to_float2(half2 v) {
    return __half22float2(v);
}


__device__ float2 cast_to_float2(float2 v) {
    return v;
}

__device__ float2 cast_to_float2(__nv_bfloat162 v) {
    return __bfloat1622float2(v);
}

__device__ void cast_from_float2(float2 v, half2& r) {
    r = __float22half2_rn(v);
}

__device__ void cast_from_float2(float2 v, float2& r) {
    r = v;
}

__device__ void cast_from_float2(float2 v, __nv_bfloat162& r) {
    r = __float22bfloat162_rn(v);
}

__device__ float2 mul2(float2 a, float2 b) {
    a.x *= b.x;
    a.y *= b.y;
    return a;
}

__device__ half2 mul2(half2 a, half2 b) {
    return __hmul2(a, b);
}


__device__ __nv_bfloat162 mul2(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __hmul2(a, b);
}



template<typename SingleVal, typename DoubleVal>
__global__ void linear_recurrence_forward_kernel_first(const SingleVal * __restrict__ a,
                               const SingleVal * __restrict__ x, float * __restrict__ s,
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
        DoubleVal next_x = *(DoubleVal *)(x + i * k);
        DoubleVal next_a = *(DoubleVal *)(a + i * k);

        float2 next_xf = cast_to_float2(next_x);
        float2 next_af = cast_to_float2(next_a);

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


template<typename SingleVal, typename DoubleVal>
__global__ void linear_recurrence_forward_kernel_third(const SingleVal * __restrict__ a,
                               const SingleVal * __restrict__ x, SingleVal* __restrict__ o, const float * __restrict__ s,
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
        DoubleVal next_x = *(DoubleVal *)(x + i * k);
        DoubleVal next_a = *(DoubleVal *)(a + i * k);

        float2 next_xf = cast_to_float2(next_x);
        float2 next_af = cast_to_float2(next_a);

        current_val.x = current_val.x * next_af.x + next_xf.x;
        current_val.y = current_val.y * next_af.y + next_xf.y;

        DoubleVal current_half;
        cast_from_float2(current_val, current_half);

        *(DoubleVal *)(o + i * k) = current_half;
    }
}


template<typename SingleVal, typename DoubleVal>
__global__ void linear_recurrence_backward_kernel_first(const SingleVal * __restrict__ a,
                               const SingleVal * __restrict__ d_o, float * __restrict__ s,
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
        DoubleVal next_d_o = *(DoubleVal *)(d_o + (n - (length_idx + i) - 1) * k);
        DoubleVal next_a = {0, 0};
        if (length_idx + i != 0) {
            next_a = *(DoubleVal *)(a + (n - (length_idx + i) - 1 + 1) * k);
        }

        float2 next_d_of = cast_to_float2(next_d_o);
        float2 next_af = cast_to_float2(next_a);

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


template<typename SingleVal, typename DoubleVal>
__global__ void linear_recurrence_backward_kernel_third(
    const SingleVal * __restrict__ a,
    const SingleVal* __restrict__ o, 
    const float * __restrict__ s,

    SingleVal * __restrict__ d_a,
    SingleVal * __restrict__ d_x, 
    const SingleVal* __restrict__ d_o, 

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
        DoubleVal next_d_o = *(DoubleVal *)(d_o + (n - (length_idx + i) - 1) * k);
        DoubleVal next_a = {0, 0};
        if (length_idx + i != 0) {
            next_a = *(DoubleVal *)(a + (n - (length_idx + i) - 1 + 1) * k);
        }

        float2 next_d_of = cast_to_float2(next_d_o);
        float2 next_af = cast_to_float2(next_a);

        current_val.x = current_val.x * next_af.x + next_d_of.x;
        current_val.y = current_val.y * next_af.y + next_d_of.y;

        DoubleVal current_half;
        cast_from_float2(current_val, current_half);

        *(DoubleVal *)(d_x + (n - (length_idx + i) - 1) * k) = current_half;


        DoubleVal current_o = {0, 0};
        if (length_idx + i != n - 1) {
            current_o = *(DoubleVal *)(o + (n - (length_idx + i) - 2) * k);    
        }

        *(DoubleVal *)(d_a + (n - (length_idx + i) - 1) * k) = mul2(current_o, current_half);
    }
}


}

template<typename SingleVal, typename DoubleVal>
void linear_recurrence_forward_cuda(
                        void *a,
                        void *x, void *o, void *s,
                        int n, int k) {
    dim3 blockDim, gridDim;
    
    blockDim.x = 32;
    blockDim.y = 8;

    gridDim.x = (n + blockDim.y * LENGTH_PER_WARP - 1) / (blockDim.y * LENGTH_PER_WARP);
    gridDim.y = (k + FEATS_PER_WARP - 1) / FEATS_PER_WARP;

    linear_recurrence_forward_kernel_first<SingleVal, DoubleVal><<<gridDim, blockDim>>>(
        (const SingleVal*) a, (const SingleVal*) x, (float*) s, n, k);

    blockDim.x = 32;
    blockDim.y = 1;

    gridDim.x = 2;
    gridDim.y = (k + FEATS_PER_WARP - 1) / FEATS_PER_WARP;

    linear_recurrence_forward_kernel_second<<<gridDim, blockDim>>>((float*) s, n, k);


    blockDim.x = 32;
    blockDim.y = 8;

    gridDim.x = (n + blockDim.y * LENGTH_PER_WARP - 1) / (blockDim.y * LENGTH_PER_WARP);
    gridDim.y = (k + FEATS_PER_WARP - 1) / FEATS_PER_WARP;

    linear_recurrence_forward_kernel_third<SingleVal, DoubleVal><<<gridDim, blockDim>>>(
        (const SingleVal*) a, (const SingleVal*) x, (SingleVal*) o, (const float*) s, n, k);
}


template<typename SingleVal, typename DoubleVal>
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

    linear_recurrence_backward_kernel_first<SingleVal, DoubleVal><<<gridDim, blockDim>>>(
        (const SingleVal*) a, (const SingleVal*) d_o, (float*) s, n, k);

    blockDim.x = 32;
    blockDim.y = 1;

    gridDim.x = 2;
    gridDim.y = (k + FEATS_PER_WARP - 1) / FEATS_PER_WARP;

    linear_recurrence_forward_kernel_second<<<gridDim, blockDim>>>((float*) s, n, k);


    blockDim.x = 32;
    blockDim.y = 8;

    gridDim.x = (n + blockDim.y * LENGTH_PER_WARP - 1) / (blockDim.y * LENGTH_PER_WARP);
    gridDim.y = (k + FEATS_PER_WARP - 1) / FEATS_PER_WARP;

    linear_recurrence_backward_kernel_third<SingleVal, DoubleVal><<<gridDim, blockDim>>>(
        (const SingleVal*) a, (const SingleVal*) o, (const float*) s, (SingleVal*) d_a, (SingleVal*) d_x, (const SingleVal*) d_o, n, k);
}

void linear_recurrence_forward_cuda_fp32(
                        void *a,
                        void *x, void *o, void* s,
                        int n, int k) {
                            linear_recurrence_forward_cuda<float, float2>(a, x, o, s, n, k);
                        }

void linear_recurrence_backward_cuda_fp32(
        void *a, void *o, void *s,
        void* d_a, void* d_x, void* d_o,
        int n, int k) {
            linear_recurrence_backward_cuda<float, float2>(a, o, s, d_a, d_x, d_o, n, k);
        }


void linear_recurrence_forward_cuda_fp16(
                        void *a,
                        void *x, void *o, void* s,
                        int n, int k) {
                            linear_recurrence_forward_cuda<half, half2>(a, x, o, s, n, k);
                        }

void linear_recurrence_backward_cuda_fp16(
        void *a, void *o, void *s,
        void* d_a, void* d_x, void* d_o,
        int n, int k) {
            linear_recurrence_backward_cuda<half, half2>(a, o, s, d_a, d_x, d_o, n, k);
        }


void linear_recurrence_forward_cuda_bf16(
                        void *a,
                        void *x, void *o, void* s,
                        int n, int k) {
                            linear_recurrence_forward_cuda<__nv_bfloat16, __nv_bfloat162>(a, x, o, s, n, k);
                        }

void linear_recurrence_backward_cuda_bf16(
        void *a, void *o, void *s,
        void* d_a, void* d_x, void* d_o,
        int n, int k) {
            linear_recurrence_backward_cuda<__nv_bfloat16, __nv_bfloat162>(a, o, s, d_a, d_x, d_o, n, k);
        }

std::vector<int> linear_recurrence_scratch_space(int n, int k) {
    int num_segments = (n + LENGTH_PER_WARP - 1) / LENGTH_PER_WARP;
    return {num_segments, 2, k}; 
}

