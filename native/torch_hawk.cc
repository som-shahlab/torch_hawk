#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>

#include "conv1d_cuda.hh"
#include "linear_recurrence_cuda.hh"

namespace {

    void* get_cuda_pointer(const pybind11::object& obj, std::string_view expected_type, const std::vector<int>& expected_shape) {
        pybind11::dict interface = obj.attr("__cuda_array_interface__");
        if (interface["version"].cast<int>() != 2) {
            throw std::invalid_argument("Unsupported Version");
        }
        if (!interface["strides"].cast<pybind11::object>().is(pybind11::none())) {
            throw std::invalid_argument("Need no strides");
        }
        if (interface["typestr"].cast<std::string>() != expected_type) {
            throw std::invalid_argument(std::string("Invalid type, expected ") + std::string(expected_type) + ", got " + std::string(pybind11::str(interface["typestr"])));
        }

        pybind11::tuple expected_shape_helper(expected_shape.size());
        for (size_t i = 0; i < expected_shape.size(); i++) {
            expected_shape_helper[i] = expected_shape[i];
        }

        if (!interface["shape"].cast<pybind11::tuple>().equal(expected_shape_helper)) {
            throw std::invalid_argument("Invalid shape");
        }

        pybind11::tuple data = interface["data"];
        return (void*)data[0].cast<uintptr_t>();
    }

    void conv1d_forward_cuda_helper(pybind11::object x, pybind11::object w, pybind11::object s, pybind11::object o, int n, int k) {
        void* x_ptr = get_cuda_pointer(x, "<f2", {n, k});
        void* w_ptr = get_cuda_pointer(w, "<f2", {k, conv1d_kernel_size()});
        void* s_ptr = get_cuda_pointer(s, "|u1", {n});
        void* o_ptr = get_cuda_pointer(o, "<f2", {n, k});
        conv1d_forward_cuda(x_ptr, w_ptr, s_ptr, o_ptr, n, k);
    }

    void conv1d_backward_cuda_helper(pybind11::object x, pybind11::object w, pybind11::object s, pybind11::object d_o, pybind11::object d_x, pybind11::object d_w, int n, int k) {
        void* x_ptr = get_cuda_pointer(x, "<f2", {n, k});
        void* w_ptr = get_cuda_pointer(w, "<f2", {k, conv1d_kernel_size()});
        void* s_ptr = get_cuda_pointer(s, "|u1", {n});
        void* d_o_ptr = get_cuda_pointer(d_o, "<f2", {n, k});
        void* d_x_ptr = get_cuda_pointer(d_x, "<f2", {n, k});
        void* d_w_ptr = get_cuda_pointer(d_w, "<f4", {k, conv1d_kernel_size()});
        conv1d_backward_cuda(x_ptr, w_ptr, s_ptr, d_o_ptr, d_x_ptr, d_w_ptr, n, k);
    }


    void linear_recurrence_forward_cuda_helper(pybind11::object a, pybind11::object x, pybind11::object o, pybind11::object s, int n, int k) {
        void* a_ptr = get_cuda_pointer(a, "<f2", {n, k});
        void* x_ptr = get_cuda_pointer(x, "<f2", {n, k});
        void* o_ptr = get_cuda_pointer(o, "<f2", {n, k});
        void* s_ptr = get_cuda_pointer(s, "<f4", linear_recurrence_scratch_space(n, k));
        linear_recurrence_forward_cuda(a_ptr, x_ptr, o_ptr, s_ptr, n, k);
    }

    void linear_recurrence_backward_cuda_helper(pybind11::object a, pybind11::object o, pybind11::object s, 
    pybind11::object d_a,  pybind11::object d_x,  pybind11::object d_o, int n, int k) {
        void* a_ptr = get_cuda_pointer(a, "<f2", {n, k});
        void* o_ptr = get_cuda_pointer(o, "<f2", {n, k});
        void* s_ptr = get_cuda_pointer(s, "<f4", linear_recurrence_scratch_space(n, k));
        void* d_a_ptr = get_cuda_pointer(d_a, "<f2", {n, k});
        void* d_x_ptr = get_cuda_pointer(d_x, "<f2", {n, k});
        void* d_o_ptr = get_cuda_pointer(d_o, "<f2", {n, k});
        linear_recurrence_backward_cuda(a_ptr, o_ptr, s_ptr, d_a_ptr, d_x_ptr, d_o_ptr, n, k);
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


    m.def("conv1d_forward_cuda", &conv1d_forward_cuda_helper, "A function that adds two numbers");
    m.def("conv1d_backward_cuda", &conv1d_backward_cuda_helper, "A function that adds two numbers");
    m.def("conv1d_kernel_size", &conv1d_kernel_size, "A function that adds two numbers");

    m.def("linear_recurrence_forward_cpu", &linear_recurrence_forward_cpu, "A function that adds two numbers");
    m.def("linear_recurrence_backward_cpu", &linear_recurrence_backward_cpu, "A function that adds two numbers");

    m.def("linear_recurrence_forward_cuda", &linear_recurrence_forward_cuda_helper, "A function that adds two numbers");
    m.def("linear_recurrence_backward_cuda", &linear_recurrence_backward_cuda_helper, "A function that adds two numbers");
    m.def("linear_recurrence_scratch_space", &linear_recurrence_scratch_space, "A function that adds two numbers");
}