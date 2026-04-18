#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
SUDOERS_DIR="${SUDOERS_DIR:-/etc/sudoers.d}"

die() {
  echo "error: $*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage:
  $0 print-sudoers <user>
  $0 install-sudoers <user>
  $0 prepare [nbds_max]
  $0 pick-free
  $0 attach <socket_path> <nbd_device>
  $0 copy-file <source_path> <dest_path>
  $0 format <nbd_device>
  $0 mountfs <nbd_device> <mount_point>
  $0 unmountfs <mount_point>
  $0 detach <nbd_device>
  $0 cleanup <mount_point> <nbd_device>
  $0 status

Commands:
  print-sudoers  Print the narrow sudoers rule for this exact script.
  install-sudoers
                 Install /etc/sudoers.d/llmdb-e2e-<user> for this script.
  prepare        Load the nbd kernel module with nbds_max (default: 16).
  pick-free      Print the first free /dev/nbdN device.
  attach         Run nbd-client -unix <socket> <device>.
  copy-file      Copy a host file into the mounted filesystem as root.
  format         Run mkfs.ext4 -F on the NBD device.
  mountfs        Mount the NBD device at the given mount point.
  unmountfs      Unmount the given mount point.
  detach         Disconnect the NBD device via nbd-client -d.
  cleanup        Unmount, then disconnect the NBD device.
  status         Print current NBD module and /dev/nbd* status.

Typical bootstrap:
  sudo $0 install-sudoers ${USER:-suero}
  sudo -n $0 prepare
  sudo -n $0 status
EOF
}

require_root() {
  [[ "$(id -u)" -eq 0 ]] || die "this command must run as root"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

require_numeric() {
  [[ "$1" =~ ^[0-9]+$ ]] || die "expected numeric value, got: $1"
}

require_existing_path() {
  [[ -e "$1" ]] || die "path does not exist: $1"
}

require_block_device() {
  [[ -b "$1" ]] || die "not a block device: $1"
}

print_sudoers() {
  local user="${1:-}"
  [[ -n "$user" ]] || die "usage: $0 print-sudoers <user>"
  printf '%s ALL=(root) NOPASSWD: %s *\n' "$user" "$SCRIPT_PATH"
}

install_sudoers() {
  require_root
  require_command visudo

  local user="${1:-}"
  [[ -n "$user" ]] || die "usage: $0 install-sudoers <user>"
  id "$user" >/dev/null 2>&1 || die "no such user: $user"

  local dest="$SUDOERS_DIR/llmdb-e2e-$user"
  local tmp
  tmp="$(mktemp)"
  trap 'rm -f "$tmp"' RETURN

  {
    printf '# Installed by %s\n' "$SCRIPT_PATH"
    print_sudoers "$user"
  } >"$tmp"

  chmod 0440 "$tmp"
  visudo -cf "$tmp" >/dev/null
  cp "$tmp" "$dest"
  chmod 0440 "$dest"

  echo "installed sudoers fragment at $dest"
  echo "verify with: sudo -n $SCRIPT_PATH status"
}

prepare() {
  require_root
  require_command modprobe

  local nbds_max="${1:-16}"
  require_numeric "$nbds_max"

  modprobe nbd "nbds_max=$nbds_max"
  status
}

pick_free() {
  local i
  for i in $(seq 0 63); do
    local dev="/dev/nbd$i"
    local pid_file="/sys/block/nbd$i/pid"
    [[ -e "$dev" ]] || continue
    if [[ ! -e "$pid_file" || ! -s "$pid_file" ]]; then
      printf '%s\n' "$dev"
      return 0
    fi
  done

  die "no free /dev/nbdN device found"
}

attach() {
  require_root
  require_command nbd-client

  local socket_path="${1:-}"
  local nbd_device="${2:-}"
  [[ -n "$socket_path" && -n "$nbd_device" ]] || die "usage: $0 attach <socket_path> <nbd_device>"

  require_existing_path "$socket_path"
  require_block_device "$nbd_device"
  nbd-client -unix "$socket_path" "$nbd_device"
}

copy_file() {
  require_root
  require_command cp

  local source_path="${1:-}"
  local dest_path="${2:-}"
  [[ -n "$source_path" && -n "$dest_path" ]] || die "usage: $0 copy-file <source_path> <dest_path>"

  require_existing_path "$source_path"
  mkdir -p "$(dirname "$dest_path")"
  cp "$source_path" "$dest_path"
}

format_device() {
  require_root
  require_command mkfs.ext4

  local nbd_device="${1:-}"
  [[ -n "$nbd_device" ]] || die "usage: $0 format <nbd_device>"

  require_block_device "$nbd_device"
  mkfs.ext4 -F "$nbd_device"
}

mountfs() {
  require_root
  require_command mount

  local nbd_device="${1:-}"
  local mount_point="${2:-}"
  [[ -n "$nbd_device" && -n "$mount_point" ]] || die "usage: $0 mountfs <nbd_device> <mount_point>"

  require_block_device "$nbd_device"
  mkdir -p "$mount_point"
  mount "$nbd_device" "$mount_point"
}

unmountfs() {
  require_root
  require_command umount

  local mount_point="${1:-}"
  [[ -n "$mount_point" ]] || die "usage: $0 unmountfs <mount_point>"

  umount "$mount_point"
}

detach() {
  require_root
  require_command nbd-client

  local nbd_device="${1:-}"
  [[ -n "$nbd_device" ]] || die "usage: $0 detach <nbd_device>"

  require_block_device "$nbd_device"
  nbd-client -d "$nbd_device"
}

cleanup() {
  require_root

  local mount_point="${1:-}"
  local nbd_device="${2:-}"
  [[ -n "$mount_point" && -n "$nbd_device" ]] || die "usage: $0 cleanup <mount_point> <nbd_device>"

  unmountfs "$mount_point"
  detach "$nbd_device"
}

status() {
  echo "script=$SCRIPT_PATH"
  echo "uid=$(id -u)"

  if [[ -e /sys/module/nbd/parameters/nbds_max ]]; then
    echo "nbd_loaded=yes"
    printf 'nbds_max='
    cat /sys/module/nbd/parameters/nbds_max
  else
    echo "nbd_loaded=no"
  fi

  local found=0
  local i
  for i in $(seq 0 63); do
    local dev="/dev/nbd$i"
    local pid_file="/sys/block/nbd$i/pid"
    [[ -e "$dev" ]] || continue
    found=1
    if [[ -e "$pid_file" && -s "$pid_file" ]]; then
      printf '%s in_use pid=%s\n' "$dev" "$(tr -d '[:space:]' <"$pid_file")"
    else
      printf '%s free\n' "$dev"
    fi
  done

  if [[ "$found" -eq 0 ]]; then
    echo "devices=none"
  fi
}

main() {
  local cmd="${1:---help}"
  shift || true

  case "$cmd" in
    --help|-h|help)
      usage
      ;;
    print-sudoers)
      print_sudoers "$@"
      ;;
    install-sudoers)
      install_sudoers "$@"
      ;;
    prepare)
      prepare "$@"
      ;;
    pick-free)
      pick_free "$@"
      ;;
    attach)
      attach "$@"
      ;;
    copy-file)
      copy_file "$@"
      ;;
    format)
      format_device "$@"
      ;;
    mountfs)
      mountfs "$@"
      ;;
    unmountfs)
      unmountfs "$@"
      ;;
    detach)
      detach "$@"
      ;;
    cleanup)
      cleanup "$@"
      ;;
    status)
      status "$@"
      ;;
    *)
      usage >&2
      die "unknown command: $cmd"
      ;;
  esac
}

main "$@"
