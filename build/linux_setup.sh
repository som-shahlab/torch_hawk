curl -o /usr/local/bin/bazel -L  https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64
chmod +x /usr/local/bin/bazel

dnf config-manager --set-enabled powertools
dnf makecache

# dnf -y install epel-release
# dnf upgrade

dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf makecache

#  dnf module install nvidia-driver:latest-dkms
dnf install cuda-toolkit

export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}

nvcc --version
