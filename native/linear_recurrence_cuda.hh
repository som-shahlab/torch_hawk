#pragma once

#include <vector>


void linear_recurrence_forward_cuda(
                        void *a,
                        void *x, void *o, void* s,
                        int n, int k);

void linear_recurrence_backward_cuda(
        void *a, void *o, void *s,
        void* d_a, void* d_x, void* d_o,
        int n, int k);

std::vector<int> linear_recurrence_scratch_space(int n, int k);