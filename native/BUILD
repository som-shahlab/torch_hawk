load(
    "@rules_foreign_cc//foreign_cc:defs.bzl", "cmake", "configure_make", "make"
)

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

load("@rules_cuda//cuda:defs.bzl", "cuda_library")


cuda_library(
    name = "conv1d_cuda",
    srcs = ["conv1d_cuda.cu"],
    hdrs = ["conv1d_cuda.hh"],
)

pybind_extension(
    name="_torch_hawk",
    srcs=[
        "torch_hawk.cc",
    ],
    linkopts = select({
        "@platforms//os:osx": ["-undefined", "dynamic_lookup"],
        "//conditions:default": [],
    }),
    deps=[
        ":conv1d_cuda",
    ],
)