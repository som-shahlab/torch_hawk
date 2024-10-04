curl -o /usr/local/bin/bazel -L  https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64
chmod +x /usr/local/bin/bazel

dnf install kernel-devel-$(uname -r) kernel-headers-$(uname -r)
dnf install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm

dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-$distro.repo

dnf clean expire-cache

dnf module install nvidia-driver:latest-dkms
dnf install cuda-toolkit

export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}

nvcc --version
