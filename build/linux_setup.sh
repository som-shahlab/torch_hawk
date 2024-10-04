curl -o /usr/local/bin/bazel -L  https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64
chmod +x /usr/local/bin/bazel


yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
# error mirrorlist.centos.org doesn't exists anymore.
sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo
yum install cuda-toolkit-12-6

#ln -s cuda-12.2 /usr/local/cuda


nvcc --version
