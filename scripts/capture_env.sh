#!/usr/bin/env bash
# Record the hardware/software environment for benchmark results.
# Usage: scripts/capture_env.sh [output_file]   (default: results/env.txt)
set -u
out="${1:-results/env.txt}"
mkdir -p "$(dirname "$out")"

section() { printf '\n## %s\n' "$1"; }
{
  echo "# pagedlm environment"
  echo "captured: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if git rev-parse --verify -q HEAD >/dev/null 2>&1; then
    echo "git commit: $(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo ' (dirty)')"
  else
    echo "git commit: none"
  fi

  section "OS"
  uname -srm
  grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '"'
  grep -qi microsoft /proc/version && echo "WSL2: yes" || echo "WSL2: no"

  section "CPU / RAM"
  lscpu | grep -E 'Model name|^CPU\(s\)'
  free -h | awk '/Mem:/ {print "RAM total: " $2}'

  section "GPU"
  nvidia-smi --query-gpu=name,driver_version,memory.total,pstate,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,power.limit,temperature.gpu \
    --format=csv
  echo "compute capability: $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)"

  section "Toolchain"
  nvcc --version | tail -2
  g++ --version | head -1
  cmake --version | head -1
  python3 --version
  python3 -m pip list 2>/dev/null | grep -iE '^(numpy|safetensors|huggingface.hub|torch|transformers) ' || true
} > "$out"

echo "wrote $out"
