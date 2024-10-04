#!/bin/sh

cd native
bazel build -c dbg _torch_hawk.so
cd ..

rm -f src/torch_hawk/_torch_hawk*
cp native/bazel-bin/_torch_hawk.so src/torch_hawk/_torch_hawk.so
