#pragma once

void conv1d_forward_cuda(
                        void *x,
                        void *w, void* s, void *o,
                        int n, int k);

void conv1d_backward_cuda(
                        void *x,
                        void *w, void* s, void *d_o,
                        void *d_x, void *d_w,
                        int n, int k);

int conv1d_kernel_size();