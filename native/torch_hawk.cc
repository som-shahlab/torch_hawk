#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>

#include "conv1d_cuda.hh"
#include "linear_recurrence_cuda.hh"

#include "dlpack.h"

namespace {

    void* get_cuda_pointer(const pybind11::object& obj, DLDataTypeCode dtype, int bits, const std::vector<int>& expected_shape) {
        pybind11::capsule cap = obj.attr("__dlpack__")();

        if (strcmp(cap.name(), "dltensor") != 0) {
            throw std::invalid_argument("Did not get tensor back");
        }

        DLManagedTensor *managed = (DLManagedTensor *)(cap.get_pointer());

        if (managed == nullptr) {
            throw std::invalid_argument("Got nullptr tensor back");

        }
        DLTensor& tensor = managed->dl_tensor;

        if (tensor.strides != nullptr) {
            int expected_stride = 1;
            for (int i = tensor.ndim - 1; i >= 0; i--) {
                if (tensor.strides[i] != expected_stride) {
                    throw std::invalid_argument("Bad stride");
                }
                expected_stride *= tensor.shape[i];
            }
        }

        if (dtype != tensor.dtype.code) {
            throw std::invalid_argument("Bad type");
        }

        if (bits != tensor.dtype.bits) {
            throw std::invalid_argument("Bad type");
        }

        if (1 != tensor.dtype.lanes) {
            throw std::invalid_argument("Bad type");
        }

        if ((int)expected_shape.size() != tensor.ndim) {
            throw std::invalid_argument("Wrong number of dimensions");
        }

        for (size_t i = 0; i < expected_shape.size(); i++) {
            if (expected_shape[i] != tensor.shape[i]) {
                throw std::invalid_argument("Invalid shape");
            }
        }

        if (tensor.byte_offset != 0) {
            throw std::invalid_argument("Bad byte offset");
        }

        return (void*)tensor.data;
    }

    void conv1d_forward_cuda_helper_fp16(pybind11::object x, pybind11::object w, pybind11::object s, pybind11::object o, int n, int k) {
        void* x_ptr = get_cuda_pointer(x, kDLFloat, 16, {n, k});
        void* w_ptr = get_cuda_pointer(w, kDLFloat, 16, {k, conv1d_kernel_size()});
        void* s_ptr = get_cuda_pointer(s, kDLUInt, 8, {n});
        void* o_ptr = get_cuda_pointer(o, kDLFloat, 16, {n, k});
        conv1d_forward_cuda_fp16(x_ptr, w_ptr, s_ptr, o_ptr, n, k);
    }

    void conv1d_backward_cuda_helper_fp16(pybind11::object x, pybind11::object w, pybind11::object s, pybind11::object d_o, pybind11::object d_x, pybind11::object d_w, int n, int k) {
        void* x_ptr = get_cuda_pointer(x, kDLFloat, 16, {n, k});
        void* w_ptr = get_cuda_pointer(w, kDLFloat, 16, {k, conv1d_kernel_size()});
        void* s_ptr = get_cuda_pointer(s, kDLUInt, 8, {n});
        void* d_o_ptr = get_cuda_pointer(d_o, kDLFloat, 16, {n, k});
        void* d_x_ptr = get_cuda_pointer(d_x, kDLFloat, 16, {n, k});
        void* d_w_ptr = get_cuda_pointer(d_w, kDLFloat, 32, {k, conv1d_kernel_size()});
        conv1d_backward_cuda_fp16(x_ptr, w_ptr, s_ptr, d_o_ptr, d_x_ptr, d_w_ptr, n, k);
    }

    void conv1d_forward_cuda_helper_bf16(pybind11::object x, pybind11::object w, pybind11::object s, pybind11::object o, int n, int k) {
        void* x_ptr = get_cuda_pointer(x, kDLBfloat, 16, {n, k});
        void* w_ptr = get_cuda_pointer(w, kDLBfloat, 16, {k, conv1d_kernel_size()});
        void* s_ptr = get_cuda_pointer(s, kDLUInt, 8, {n});
        void* o_ptr = get_cuda_pointer(o, kDLBfloat, 16, {n, k});
        conv1d_forward_cuda_bf16(x_ptr, w_ptr, s_ptr, o_ptr, n, k);
    }

    void conv1d_backward_cuda_helper_bf16(pybind11::object x, pybind11::object w, pybind11::object s, pybind11::object d_o, pybind11::object d_x, pybind11::object d_w, int n, int k) {
        void* x_ptr = get_cuda_pointer(x, kDLBfloat, 16, {n, k});
        void* w_ptr = get_cuda_pointer(w, kDLBfloat, 16, {k, conv1d_kernel_size()});
        void* s_ptr = get_cuda_pointer(s, kDLUInt, 8, {n});
        void* d_o_ptr = get_cuda_pointer(d_o, kDLBfloat, 16, {n, k});
        void* d_x_ptr = get_cuda_pointer(d_x, kDLBfloat, 16, {n, k});
        void* d_w_ptr = get_cuda_pointer(d_w, kDLFloat, 32, {k, conv1d_kernel_size()});
        conv1d_backward_cuda_bf16(x_ptr, w_ptr, s_ptr, d_o_ptr, d_x_ptr, d_w_ptr, n, k);
    }


    void conv1d_forward_cuda_helper_fp32(pybind11::object x, pybind11::object w, pybind11::object s, pybind11::object o, int n, int k) {
        void* x_ptr = get_cuda_pointer(x, kDLFloat, 32, {n, k});
        void* w_ptr = get_cuda_pointer(w, kDLFloat, 32, {k, conv1d_kernel_size()});
        void* s_ptr = get_cuda_pointer(s, kDLUInt, 8, {n});
        void* o_ptr = get_cuda_pointer(o, kDLFloat, 32, {n, k});
        conv1d_forward_cuda_fp32(x_ptr, w_ptr, s_ptr, o_ptr, n, k);
    }

    void conv1d_backward_cuda_helper_fp32(pybind11::object x, pybind11::object w, pybind11::object s, pybind11::object d_o, pybind11::object d_x, pybind11::object d_w, int n, int k) {
        void* x_ptr = get_cuda_pointer(x, kDLFloat, 32, {n, k});
        void* w_ptr = get_cuda_pointer(w, kDLFloat, 32, {k, conv1d_kernel_size()});
        void* s_ptr = get_cuda_pointer(s, kDLUInt, 8, {n});
        void* d_o_ptr = get_cuda_pointer(d_o, kDLFloat, 32, {n, k});
        void* d_x_ptr = get_cuda_pointer(d_x, kDLFloat, 32, {n, k});
        void* d_w_ptr = get_cuda_pointer(d_w, kDLFloat, 32, {k, conv1d_kernel_size()});
        conv1d_backward_cuda_fp32(x_ptr, w_ptr, s_ptr, d_o_ptr, d_x_ptr, d_w_ptr, n, k);
    }


    void linear_recurrence_forward_cuda_helper_fp16(pybind11::object a, pybind11::object x, pybind11::object o, pybind11::object s, int n, int k) {
        void* a_ptr = get_cuda_pointer(a, kDLFloat, 16, {n, k});
        void* x_ptr = get_cuda_pointer(x, kDLFloat, 16, {n, k});
        void* o_ptr = get_cuda_pointer(o, kDLFloat, 16, {n, k});
        void* s_ptr = get_cuda_pointer(s, kDLFloat, 32, linear_recurrence_scratch_space(n, k));
        linear_recurrence_forward_cuda_fp16(a_ptr, x_ptr, o_ptr, s_ptr, n, k);
    }

    void linear_recurrence_backward_cuda_helper_fp16(pybind11::object a, pybind11::object o, pybind11::object s, 
    pybind11::object d_a,  pybind11::object d_x,  pybind11::object d_o, int n, int k) {
        void* a_ptr = get_cuda_pointer(a, kDLFloat, 16, {n, k});
        void* o_ptr = get_cuda_pointer(o, kDLFloat, 16, {n, k});
        void* s_ptr = get_cuda_pointer(s, kDLFloat, 32, linear_recurrence_scratch_space(n, k));
        void* d_a_ptr = get_cuda_pointer(d_a, kDLFloat, 16, {n, k});
        void* d_x_ptr = get_cuda_pointer(d_x, kDLFloat, 16, {n, k});
        void* d_o_ptr = get_cuda_pointer(d_o, kDLFloat, 16, {n, k});
        linear_recurrence_backward_cuda_fp16(a_ptr, o_ptr, s_ptr, d_a_ptr, d_x_ptr, d_o_ptr, n, k);
    }

    void linear_recurrence_forward_cuda_helper_bf16(pybind11::object a, pybind11::object x, pybind11::object o, pybind11::object s, int n, int k) {
        void* a_ptr = get_cuda_pointer(a, kDLBfloat, 16, {n, k});
        void* x_ptr = get_cuda_pointer(x, kDLBfloat, 16, {n, k});
        void* o_ptr = get_cuda_pointer(o, kDLBfloat, 16, {n, k});
        void* s_ptr = get_cuda_pointer(s, kDLFloat, 32, linear_recurrence_scratch_space(n, k));
        linear_recurrence_forward_cuda_bf16(a_ptr, x_ptr, o_ptr, s_ptr, n, k);
    }

    void linear_recurrence_backward_cuda_helper_bf16(pybind11::object a, pybind11::object o, pybind11::object s, 
    pybind11::object d_a,  pybind11::object d_x,  pybind11::object d_o, int n, int k) {
        void* a_ptr = get_cuda_pointer(a, kDLBfloat, 16, {n, k});
        void* o_ptr = get_cuda_pointer(o, kDLBfloat, 16, {n, k});
        void* s_ptr = get_cuda_pointer(s, kDLFloat, 32, linear_recurrence_scratch_space(n, k));
        void* d_a_ptr = get_cuda_pointer(d_a, kDLBfloat, 16, {n, k});
        void* d_x_ptr = get_cuda_pointer(d_x, kDLBfloat, 16, {n, k});
        void* d_o_ptr = get_cuda_pointer(d_o, kDLBfloat, 16, {n, k});
        linear_recurrence_backward_cuda_bf16(a_ptr, o_ptr, s_ptr, d_a_ptr, d_x_ptr, d_o_ptr, n, k);
    }



    void linear_recurrence_forward_cuda_helper_fp32(pybind11::object a, pybind11::object x, pybind11::object o, pybind11::object s, int n, int k) {
        void* a_ptr = get_cuda_pointer(a, kDLFloat, 32, {n, k});
        void* x_ptr = get_cuda_pointer(x, kDLFloat, 32, {n, k});
        void* o_ptr = get_cuda_pointer(o, kDLFloat, 32, {n, k});
        void* s_ptr = get_cuda_pointer(s, kDLFloat, 32, linear_recurrence_scratch_space(n, k));
        linear_recurrence_forward_cuda_fp32(a_ptr, x_ptr, o_ptr, s_ptr, n, k);
    }

    void linear_recurrence_backward_cuda_helper_fp32(pybind11::object a, pybind11::object o, pybind11::object s, 
    pybind11::object d_a,  pybind11::object d_x,  pybind11::object d_o, int n, int k) {
        void* a_ptr = get_cuda_pointer(a, kDLFloat, 32, {n, k});
        void* o_ptr = get_cuda_pointer(o, kDLFloat, 32, {n, k});
        void* s_ptr = get_cuda_pointer(s, kDLFloat, 32, linear_recurrence_scratch_space(n, k));
        void* d_a_ptr = get_cuda_pointer(d_a, kDLFloat, 32, {n, k});
        void* d_x_ptr = get_cuda_pointer(d_x, kDLFloat, 32, {n, k});
        void* d_o_ptr = get_cuda_pointer(d_o, kDLFloat, 32, {n, k});
        linear_recurrence_backward_cuda_fp32(a_ptr, o_ptr, s_ptr, d_a_ptr, d_x_ptr, d_o_ptr, n, k);
    }

    void conv1d_forward_cpu(pybind11::array_t<float> x, pybind11::array_t<float> w, pybind11::array_t<uint8_t> s, pybind11::array_t<float> o, int n, int k) {
        for (int ni = 0; ni < n; ni++) {
            for (int ki = 0; ki < k; ki++) {
                float result = 0;
                for (int j = 0; j < conv1d_kernel_size(); j++) {
                    int index = ni - conv1d_kernel_size() + 1 + j;
                    if (index >= 0 && (s.at(index) == s.at(ni))) {
                        result += x.at(index, ki) * w.at(ki, j);
                    }
                }
                o.mutable_at(ni, ki) = result;
            }
        }
    }

    void conv1d_backward_cpu(pybind11::array_t<float> x, pybind11::array_t<float> w, pybind11::array_t<uint8_t> s, pybind11::array_t<float> d_o, pybind11::array_t<float> d_x, pybind11::array_t<float> d_w, int n, int k) {
        for (int ki = 0; ki < k; ki++) {
            for (int j = 0; j < conv1d_kernel_size(); j++) {
                d_w.mutable_at(ki, j) = 0;
            }
        }

        for (int ni = 0; ni < n; ni++) {
            for (int ki = 0; ki < k; ki++) {
                float result = 0;
                for (int j = 0; j < conv1d_kernel_size(); j++) {
                    int index = ni + j;
                    if (index < n && (s.at(index) == s.at(ni))) {
                        result += d_o.at(index, ki) * w.at(ki, conv1d_kernel_size() - j - 1);
                        d_w.mutable_at(ki, conv1d_kernel_size() - j - 1) += d_o.at(index, ki) * x.at(ni, ki);
                    }
                }
                d_x.mutable_at(ni, ki) = result;
            }
        }
    }

    void linear_recurrence_forward_cpu(pybind11::array_t<float> a, pybind11::array_t<float> x, pybind11::array_t<float> o, int n, int k) {
        for (int ki = 0; ki < k; ki++) {
            float current = 0;
            for (int ni = 0; ni < n; ni++) {
                current = current * a.at(ni, ki) + x.at(ni, ki);
                o.mutable_at(ni, ki) = current;
            }
        }
    }

    void linear_recurrence_backward_cpu(
        pybind11::array_t<float> a, pybind11::array_t<float> o, 
        pybind11::array_t<float> d_a, pybind11::array_t<float> d_x, pybind11::array_t<float> d_o,
        int n, int k
    ) {
        for (int ki = 0; ki < k; ki++) {
            d_a.mutable_at(0, ki) = 0;
            
            float current = d_o.at(n - 1, ki);
            for (int ni = n - 1; ni >= 0; ni--) {
                d_x.mutable_at(ni, ki) = current;
                if (ni != 0) {
                    d_a.mutable_at(ni, ki) = o.at(ni - 1, ki) * current;
                    current = current * a.at(ni, ki) + d_o.at(ni - 1, ki);
                }
            }
        }
    }

}  // namespace

PYBIND11_MODULE(_torch_hawk, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("conv1d_forward_cpu", &conv1d_forward_cpu, "A function that adds two numbers");
    m.def("conv1d_backward_cpu", &conv1d_backward_cpu, "A function that adds two numbers");


    m.def("conv1d_forward_cuda_fp32", &conv1d_forward_cuda_helper_fp32, "A function that adds two numbers");
    m.def("conv1d_backward_cuda_fp32", &conv1d_backward_cuda_helper_fp32, "A function that adds two numbers");
    

    m.def("conv1d_forward_cuda_fp16", &conv1d_forward_cuda_helper_fp16, "A function that adds two numbers");
    m.def("conv1d_backward_cuda_fp16", &conv1d_backward_cuda_helper_fp16, "A function that adds two numbers");

    m.def("conv1d_forward_cuda_bf16", &conv1d_forward_cuda_helper_bf16, "A function that adds two numbers");
    m.def("conv1d_backward_cuda_bf16", &conv1d_backward_cuda_helper_bf16, "A function that adds two numbers");


    m.def("conv1d_kernel_size", &conv1d_kernel_size, "A function that adds two numbers");

    m.def("linear_recurrence_forward_cpu", &linear_recurrence_forward_cpu, "A function that adds two numbers");
    m.def("linear_recurrence_backward_cpu", &linear_recurrence_backward_cpu, "A function that adds two numbers");

    m.def("linear_recurrence_forward_cuda_fp32", &linear_recurrence_forward_cuda_helper_fp32, "A function that adds two numbers");
    m.def("linear_recurrence_backward_cuda_fp32", &linear_recurrence_backward_cuda_helper_fp32, "A function that adds two numbers");
   
    m.def("linear_recurrence_forward_cuda_fp16", &linear_recurrence_forward_cuda_helper_fp16, "A function that adds two numbers");
    m.def("linear_recurrence_backward_cuda_fp16", &linear_recurrence_backward_cuda_helper_fp16, "A function that adds two numbers");
    
    m.def("linear_recurrence_forward_cuda_bf16", &linear_recurrence_forward_cuda_helper_bf16, "A function that adds two numbers");
    m.def("linear_recurrence_backward_cuda_bf16", &linear_recurrence_backward_cuda_helper_bf16, "A function that adds two numbers");
    
    m.def("linear_recurrence_scratch_space", &linear_recurrence_scratch_space, "A function that adds two numbers");
}