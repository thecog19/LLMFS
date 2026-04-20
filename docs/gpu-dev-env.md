# GPU development environment (WSL2 + NVIDIA via wgpu)

LLMDB's V2+ compute path targets wgpu. On WSL2 with NVIDIA hardware
the wgpu → Vulkan stack needs Mesa's `dzn` (Dozen) Vulkan-over-D3D12
ICD, because NVIDIA ships no Linux Vulkan driver for WSL and
Ubuntu's `mesa-vulkan-drivers` package doesn't enable dzn.

This document captures the one-time setup needed to get
`wgpu::Backends::VULKAN` to enumerate and drive the GPU. Validated
against:
- WSL2 kernel `6.6.87.2-microsoft-standard-WSL2`
- Ubuntu 24.04 (noble)
- NVIDIA driver 572.90 (Windows host) / CUDA 12.8 exposed via
  `/dev/dxg`
- `GeForce RTX 5070 Laptop GPU` (sm_120, 36 SMs)

## Prerequisites check

Confirm these before starting:

```sh
ls /dev/dxg                     # must exist — WSL GPU passthrough
nvidia-smi                      # must work — NVIDIA WSL CUDA driver installed
ls /usr/lib/wsl/lib/libd3d12.so # must exist — D3D12 translation lib
```

If any of those are missing, install the NVIDIA "CUDA on WSL"
driver on the Windows side first:
<https://developer.nvidia.com/cuda/wsl>.

## Install build dependencies

```sh
sudo apt install -y build-essential meson ninja-build python3-mako \
    pkg-config bison flex libexpat1-dev directx-headers-dev \
    spirv-tools glslang-tools git libdrm-dev libzstd-dev vulkan-tools

# Ubuntu 24.04 ships meson 1.3.2; Mesa main requires >= 1.4. Upgrade
# in user-space (doesn't touch system meson):
pip3 install --user --break-system-packages --upgrade meson
export PATH="$HOME/.local/bin:$PATH"
meson --version   # expect 1.11.x or newer
```

## Build Mesa dzn

Only the `microsoft-experimental` Vulkan driver is built; everything
else is disabled. Takes a handful of minutes on a modern CPU.

```sh
cd /tmp
git clone --depth 1 https://gitlab.freedesktop.org/mesa/mesa.git mesa-src
cd mesa-src

meson setup build-dzn \
    -Dvulkan-drivers=microsoft-experimental \
    -Dgallium-drivers= \
    -Dplatforms= \
    -Dglx=disabled -Degl=disabled -Dgles1=disabled -Dgles2=disabled \
    -Dopengl=false -Dshared-glapi=disabled -Dvalgrind=disabled \
    -Dllvm=disabled -Dtools=

ninja -C build-dzn
```

Artifacts land at:
- `build-dzn/src/microsoft/vulkan/libvulkan_dzn.so`
- `build-dzn/src/microsoft/spirv_to_dxil/libspirv_to_dxil.so`
- `build-dzn/src/microsoft/vulkan/dzn_icd.x86_64.json` (template; paths
  point to `/usr/local/lib/...` which we rewrite)

## Install to user-space

No `sudo make install` — everything lives under `$HOME`:

```sh
mkdir -p ~/.local/lib/dzn ~/.local/share/vulkan/icd.d
cp /tmp/mesa-src/build-dzn/src/microsoft/vulkan/libvulkan_dzn.so ~/.local/lib/dzn/
cp /tmp/mesa-src/build-dzn/src/microsoft/spirv_to_dxil/libspirv_to_dxil.so ~/.local/lib/dzn/

cat > ~/.local/share/vulkan/icd.d/dzn_icd.x86_64.json <<EOF
{
    "ICD": {
        "api_version": "1.1.348",
        "library_arch": "64",
        "library_path": "$HOME/.local/lib/dzn/libvulkan_dzn.so"
    },
    "file_format_version": "1.0.1"
}
EOF
```

## Runtime environment

Every shell that runs llmdb binaries with GPU access needs:

```sh
export VK_DRIVER_FILES="$HOME/.local/share/vulkan/icd.d/dzn_icd.x86_64.json"
export LD_LIBRARY_PATH="$HOME/.local/lib/dzn:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
```

Verify with `vulkaninfo --summary`. Expect a `GPU0:` entry with
`deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`, `driverID =
DRIVER_ID_MESA_DOZEN`, and `deviceName = "Microsoft Direct3D12
(NVIDIA ...)"`.

Add those two lines to `~/.bashrc` if you want them sticky.

## wgpu-side requirement

dzn reports `conformanceVersion = 0.0.0.0` because it's a translation
layer, not a Khronos-certified driver. wgpu filters non-conformant
adapters by default. Bypass:

```rust
let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
    backends: wgpu::Backends::VULKAN,
    flags: wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
    ..Default::default()
});
```

The LLMDB wgpu-using modules set this flag centrally. If a new
scratch binary skips it, wgpu will report "only CPU rasterizers
visible" — that's the symptom.

## Known limits on this stack

- `max_storage_buffer_binding_size`: 128 MiB. Large tensors need
  splitting across multiple bindings.
- `max_compute_invocations_per_workgroup`: 1024.
- `conformanceVersion`: 0.0.0.0 — some Vulkan features that assume
  full conformance may misbehave. Stick to compute-only paths (no
  graphics) and validate against a CPU reference (see L0 in
  DESIGN-NEW §15).

## Pipeline (what actually runs)

```
our WGSL kernel
  → wgpu
    → Vulkan loader
      → dzn (libvulkan_dzn.so)
        → D3D12 runtime (libd3d12.so from /usr/lib/wsl/lib/)
          → NVIDIA Windows driver
            → RTX 5070 Laptop GPU
```

Two translation layers: Vulkan→D3D12 (dzn) and D3D12→NVIDIA driver
(Microsoft DirectX). Latency and non-conformance edge cases live
in those layers. Acceptable for development; if this ever bottlenecks
in practice, the fallback is `cudarc` + CUDA kernels, which takes
the direct driver path.

## Rebuild

Mesa updates land constantly; if dzn needs updating:

```sh
cd /tmp/mesa-src
git pull --depth 1
ninja -C build-dzn
cp build-dzn/src/microsoft/vulkan/libvulkan_dzn.so ~/.local/lib/dzn/
cp build-dzn/src/microsoft/spirv_to_dxil/libspirv_to_dxil.so ~/.local/lib/dzn/
```

Shared object paths in the JSON manifest don't need to change.
