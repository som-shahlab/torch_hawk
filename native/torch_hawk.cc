#include <pybind11/pybind11.h>

#include <iostream>

#include "conv1d_cuda.hh"

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
            throw std::invalid_argument("Invalid type");
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
        void* w_ptr = get_cuda_pointer(w, "<f2", {k, 4});
        void* s_ptr = get_cuda_pointer(s, "<f2", {n});
        void* o_ptr = get_cuda_pointer(o, "<f2", {n, k});
        conv1d_forward_cuda(x_ptr, w_ptr, s_ptr, o_ptr, n, k);
    }

}  // namespace

PYBIND11_MODULE(_torch_hawk, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("conv1d_forward_cuda", &conv1d_forward_cuda_helper, "A function that adds two numbers");
}