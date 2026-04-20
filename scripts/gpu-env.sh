#!/usr/bin/env bash
# Source this (`. scripts/gpu-env.sh`) before running any llmdb
# binary that needs GPU access via the wgpu/Vulkan/dzn path on WSL2.
#
# Sets the two env vars that tell the Vulkan loader to use the
# user-space Mesa dzn ICD and to resolve D3D12 translation libs
# out of /usr/lib/wsl/lib/. See docs/gpu-dev-env.md for the one-
# time setup that populates ~/.local/lib/dzn and the ICD manifest.

if [ ! -f "$HOME/.local/share/vulkan/icd.d/dzn_icd.x86_64.json" ]; then
    echo "dzn ICD manifest missing at ~/.local/share/vulkan/icd.d/dzn_icd.x86_64.json" >&2
    echo "See docs/gpu-dev-env.md for the Mesa dzn build + install steps." >&2
    return 1 2>/dev/null || exit 1
fi

export VK_DRIVER_FILES="$HOME/.local/share/vulkan/icd.d/dzn_icd.x86_64.json"
export LD_LIBRARY_PATH="$HOME/.local/lib/dzn:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
