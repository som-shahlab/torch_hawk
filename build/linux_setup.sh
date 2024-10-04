curl -o /usr/local/bin/bazel -L  https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64
chmod +x /usr/local/bin/bazel

wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run
sh cuda_12.6.2_560.35.03_linux.run --silent --toolkit

nvcc --version