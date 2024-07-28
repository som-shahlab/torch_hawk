#pragma once

void conv1d_forward_cuda(
                        void *x,
                        void *w, void* s, void *o,
                        int n, int k);