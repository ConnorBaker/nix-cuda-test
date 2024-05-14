#!/usr/bin/env bash

set -euo pipefail

declare -r USER="runner"
declare -r USER_HOME="/home/$USER"
declare -r BTRFS_BLOCK_DEVICE="/dev/nvme0n1"
declare -r BTRFS_MOUNT="/mnt/fs"

_log() {
  if (($# != 2)); then
    echo "_log: missing function name and message" >&2
    exit 1
  fi
  echo "[$(date)][${1:?}] ${2:?}"
}

installPrerequisites() {
  log() { _log "installPrerequisites" "$@"; }
  local -ar packages=(
    # Btrfs
    "btrfs-progs"
    "gdisk"
    # bpftune
    # See https://github.com/oracle/bpftune?tab=readme-ov-file#getting-started
    "make"
    "libbpf1"
    "libbpf-dev"
    "libcap-dev"
    "linux-tools-common" # Provides bpftool
    "libnl-route-3-dev"  # TODO: This wasn't one of the dependencies listed in the README
    "libnl-3-dev"
    "clang"
    "llvm"
    "python3-docutils"
    # ZRAM
    "linux-modules-extra-azure"
    "zstd"
    # Generally required
    "git"
    "gpg"
    "inetutils-ping"
  )

  log "Updating apt"
  sudo apt-get update

  log "Installing packages: ${packages[*]}"
  sudo apt-get install -y "${packages[@]}"
}

setupZramSwap() {
  log() { _log "setupZramSwap" "$@"; }
  local -r swapSize="1TB"

  log "Enabling zram module"
  sudo modprobe zram num_devices=1

  log "Creating zram0 device"
  sudo zramctl --find --size "$swapSize" --algorithm zstd

  log "Enabling zram0 device"
  sudo mkswap /dev/zram0
  sudo swapon --priority -2 /dev/zram0
}

setupBtrfsMntFsVolume() {
  log() { _log "setupBtrfsMntFsVolume" "$@"; }
  local -ar disks=(
    "/dev/nvme0n1"
    "/dev/nvme1n1"
  )

  log "Creating Btrfs volume"
  for disk in "${disks[@]}"; do
    log "Processing $disk"

    log "Wiping disk"
    sudo sgdisk --zap-all "$disk"

    log "Creating GPT"
    sudo parted --script "$disk" mklabel gpt mkpart primary 0% 100%
  done

  log "Waiting for device nodes to appear"
  sudo udevadm settle

  log "Formatting disks"
  sudo mkfs.btrfs --force --label fs --data raid0 "${disks[@]}"

  log "Mounting disks"
  sudo mkdir -p "$BTRFS_MOUNT"
  sudo mount -t btrfs -o defaults,noatime "$BTRFS_BLOCK_DEVICE" "$BTRFS_MOUNT"
}

createBtrfsMntFsSubvolume() {
  log() { _log "createBtrfsMntFsSubvolume" "$@"; }
  if (($# != 2)); then
    log "!!! missing subvolume name and path !!!" >&2
    exit 1
  fi
  local -r name="$1"
  local -r mountPoint="$2"

  log "Creating subvolume $name"
  sudo btrfs subvolume create "$BTRFS_MOUNT/$name"

  log "Mounting subvolume $name"
  sudo mkdir -p "$mountPoint" "$BTRFS_MOUNT/$name"
  sudo mount -t btrfs -o defaults,noatime,subvol="$name" "$BTRFS_BLOCK_DEVICE" "$mountPoint"
}

setupBtrfsMntFsSubvolumes() {
  log() { _log "setupBtrfsMntFsSubvolumes" "$@"; }
  local -ar subvolumeNames=(
    "nix"
    "tmp"
    "working"
  )

  log "Creating Btrfs subvolumes"
  for name in "${subvolumeNames[@]}"; do
    createBtrfsMntFsSubvolume "$name" "/$name"
  done

  log "Fixing permissions on /tmp"
  sudo chmod -R 1777 "/tmp"

  log "Fixing permissions on /working"
  sudo chown -R "$USER:$USER" "/working"

  log "Setting up an OverlayFS mount for $USER_HOME on /working"
  local -r lowerDir="/home/.$USER"
  local -r upperDir="/working/$USER"
  local -r workDir="/working/.$USER"
  sudo mv "$USER_HOME" "$lowerDir"
  sudo mkdir -p "$upperDir" "$workDir" "$USER_HOME"
  sudo mount -t overlay overlay -o lowerdir="$lowerDir",upperdir="$upperDir",workdir="$workDir" "$USER_HOME"

  log "Setting permissions on $USER_HOME"
  sudo chown -R "$USER:$USER" "$USER_HOME"
}

setupWarp() {
  log() { _log "setupWarp" "$@"; }
  local -r BASE_URL="https://pkg.cloudflareclient.com"
  local -r GPG_KEY_URL="$BASE_URL/pubkey.gpg"
  local -r CLOUDFLARE_KEYRING="/usr/share/keyrings/cloudflare-archive-keyring.gpg"
  local -r CLOUDFLARE_REPO="/etc/apt/sources.list.d/cloudflare-client.list"

  log "Adding Cloudflare GPG key"
  curl --location "$GPG_KEY_URL" | sudo gpg --yes --dearmor --output "$CLOUDFLARE_KEYRING"

  # TODO: Using Jammy as 23.xx releases aren't supported by Cloudflare yet.
  log "Adding Cloudflare repository"
  log "!!! Using Jammy as 23.xx releases aren't supported by Cloudflare yet !!!"
  echo "deb [signed-by=$CLOUDFLARE_KEYRING] $BASE_URL/ jammy main" | sudo tee "$CLOUDFLARE_REPO"

  log "Updating apt"
  sudo apt update

  log "Installing Cloudflare Warp"
  sudo apt install -y cloudflare-warp

  log "Starting Cloudflare Warp"
  sudo warp-cli --accept-tos registration new
  sudo warp-cli --accept-tos mode warp+doh
  sudo warp-cli --accept-tos connect

  log "Testing IPv4 connectivity"
  curl --ipv4 --silent --max-time 15 --retry 3 --user-agent Mozilla https://api.ip.sb/geoip

  log "Testing IPv6 connectivity"
  curl --ipv6 --silent --max-time 15 --retry 3 --user-agent Mozilla https://api.ip.sb/geoip
}

setupNix() {
  log() { _log "setupNix" "$@"; }
  local -r NIX_CONFIG="/etc/nix/nix.conf"
  local -r NIX_INSTALLER="https://install.determinate.systems/nix"
  local -ra extraConfig=(
    "accept-flake-config = true"
    "allow-import-from-derivation = false"
    "auto-allocate-uids = true"
    "builders-use-substitutes = true"
    "connect-timeout = 10"
    "experimental-features = auto-allocate-uids cgroups fetch-closure fetch-tree flakes git-hashing nix-command no-url-literals parse-toml-timestamps verified-fetches"
    "fsync-metadata = false"
    "http-connections = 256"
    "log-lines = 100"
    "max-substitution-jobs = 128"
    "narinfo-cache-negative-ttl = 0"
    "sandbox-fallback = false"
    "substituters = https://cantcache.me/cuda https://cache.nixos.org"
    "trusted-public-keys = cuda:NtbpAU7XGYlttrhCduqvpYKottCPdWVITWT+3nFVTBY= cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
    "trusted-substituters = https://cantcache.me/cuda https://cuda-maintainers.cachix.org"
    "trusted-users = root runner @wheel"
    "use-cgroups = true"
    "use-xdg-base-directories = true"
    "warn-dirty = false"
  )

  log "Installing Nix"
  curl --proto '=https' --tlsv1.2 -sSf -L "$NIX_INSTALLER" |
    sh -s -- install --no-confirm

  log "Adding extra Nix configuration"
  for line in "${extraConfig[@]}"; do
    echo "$line" | sudo tee -a "$NIX_CONFIG"
  done

  log "Overriding defaults in Nix configuration"
  sudo sed \
    -e 's/auto-optimise-store = true/auto-optimise-store = false/g' \
    -e 's/build-users-group = nixbld/build-users-group =/g' \
    -i "$NIX_CONFIG"

  log "Reloading Nix configuration"
  sudo systemctl reload nix-daemon
}

setupAttic() {
  log() { _log "setupAttic" "$@"; }
  local -r ATTIC_USER="attic-watch-store"
  local -r ATTIC_SERVICE_HOME="/var/lib/$ATTIC_USER"
  local -r REV="4dbdbee45728d8ce5788db6461aaaa89d98081f0"

  log "Installing Attic"
  nix profile install --print-build-logs "github:zhaofengli/attic/$REV#attic"

  log "Creating a user for the attic service"
  sudo useradd --system --shell /usr/sbin/nologin --home-dir "$ATTIC_SERVICE_HOME" "$ATTIC_USER"

  log "Creating directories for the attic service"

  # TODO
}

setupKernelVmParameters() {
  log() { _log "setupKernelVmParameters" "$@"; }

  # Taken from: https://github.com/ConnorBaker/nixos-configs/blob/e6d3e54ed9d257bd148a5bfb57dc476570b5d9f0/modules/zram.nix
  local -ra vmParameters=(
    # https://wiki.archlinux.org/title/Zram#Optimizing_swap_on_zram
    "vm.watermark_boost_factor=0"
    "vm.watermark_scale_factor=125"
    "vm.page-cluster=0"

    # https://github.com/pop-os/default-settings/blob/master_noble/etc/sysctl.d/10-pop-default-settings.conf
    "vm.swappiness=190" # Strong preference for ZRAM
    "vm.max_map_count=2147483642"

    # Higher values since these machines are used mostly as remote builders
    "vm.dirty_ratio=80"
    "vm.dirty_background_ratio=50"
  )

  log "Setting up kernel VM parameters"
  for param in "${vmParameters[@]}"; do
    sudo sysctl -w "$param"
  done
}

setupKernelNetParameters() {
  log() { _log "setupKernelNetParameters" "$@"; }

  # Taken from: https://github.com/ConnorBaker/nixos-configs/blob/e6d3e54ed9d257bd148a5bfb57dc476570b5d9f0/modules/networking.nix
  local -ri KB=1024
  local -ri MB=$((KB * KB))

  # Memory settings
  local -ri memMin=$((8 * KB))
  local -ri rmemDefault=$((128 * KB))
  local -ri wmemDefault=$((16 * KB))
  local -ri memMax=$((16 * MB))

  local -ra netParameters=(
    # Enable BPF JIT for better performance
    "net.core.bpf_jit_enable=1"
    "net.core.bpf_jit_harden=0"

    # Change the default queueing discipline to cake and the congestion control algorithm to BBR
    "net.core.default_qdisc=cake"
    "net.ipv4.tcp_congestion_control=bbr"

    # Largely taken from https://wiki.archlinux.org/title/sysctl and
    # https://github.com/redhat-performance/tuned/blob/master/profiles/network-throughput/tuned.conf#L10
    "net.core.somaxconn=$((8 * KB))"
    "net.core.netdev_max_backlog=$((16 * KB))"
    "net.core.optmem_max=$((64 * KB))"

    # RMEM
    "net.core.rmem_default=$rmemDefault"
    "net.core.rmem_max=$memMax"
    "net.ipv4.tcp_rmem=$memMin $rmemDefault $memMax"
    "net.ipv4.udp_rmem_min=$memMin"

    # WMEM
    "net.core.wmem_default=$wmemDefault"
    "net.core.wmem_max=$memMax"
    "net.ipv4.tcp_wmem=$memMin $wmemDefault $memMax"
    "net.ipv4.udp_wmem_min=$memMin"

    # General TCP
    "net.ipv4.tcp_fastopen=3"
    "net.ipv4.tcp_fin_timeout=10"
    "net.ipv4.tcp_keepalive_intvl=10"
    "net.ipv4.tcp_keepalive_probes=6"
    "net.ipv4.tcp_keepalive_time=60"
    "net.ipv4.tcp_max_syn_backlog=$((8 * KB))"
    "net.ipv4.tcp_max_tw_buckets=2000000"
    "net.ipv4.tcp_mtu_probing=1"
    "net.ipv4.tcp_slow_start_after_idle=0"
    "net.ipv4.tcp_tw_reuse=1"
  )

  log "Setting up kernel network parameters"
  for param in "${netParameters[@]}"; do
    sudo sysctl -w "$param"
  done
}

setupBpftune() {
  log() { _log "setupBpftune" "$@"; }
  local -r BASE_URL="https://github.com/oracle/bpftune/archive"
  local -r REV="0e6bca2e5880fcbaac6478c4042f5f9314e61463"
  local -r TARBALL_NAME="bpftune-$REV.tar.gz"
  local -r BPFTUNE_DIR="$USER_HOME/bpftune"

  log "Creating directory for bpftune"
  mkdir -p "$BPFTUNE_DIR"

  log "Entering directory for bpftune"
  pushd "$BPFTUNE_DIR"

  log "Downloading bpftune tarball"
  curl --location "$BASE_URL/$REV.tar.gz" --output "$TARBALL_NAME"

  log "Extracting bpftune tarball"
  tar xzf "$TARBALL_NAME" --strip-components=1

  log "Removing downloaded archive"
  rm -f "$TARBALL_NAME"

  log "Building bpftune"
  make -j

  log "Installing bpftune"
  sudo make install

  log "Staring bpftune"
  sudo systemctl enable bpftune
  sudo systemctl start bpftune

  log "Exiting directory for bpftune"
  popd
}

setupGitHubActionsRunner() {
  log() { _log "setupGitHubActionsRunner" "$@"; }
  local -r BASE_URL="https://github.com/actions/runner/releases/download"
  local -r RELEASE="2.316.1"
  local -r SHA256="d62de2400eeeacd195db91e2ff011bfb646cd5d85545e81d8f78c436183e09a8"
  local -r TARBALL_NAME="actions-runner-linux-x64-$RELEASE.tar.gz"
  local -r ACTIONS_RUNNER_DIR="$USER_HOME/actions-runner"

  log "Creating directory for GitHub Actions Runner"
  mkdir -p "$ACTIONS_RUNNER_DIR"

  log "Entering directory for GitHub Actions Runner"
  pushd "$ACTIONS_RUNNER_DIR"

  log "Downloading GitHub Actions Runner"
  curl --location "$BASE_URL/v$RELEASE/$TARBALL_NAME" --output "$TARBALL_NAME"

  log "Verifying SHA256 checksum"
  echo "$SHA256 $TARBALL_NAME" | sha256sum --check

  log "Extracting GitHub Actions Runner"
  tar xzf "$TARBALL_NAME"

  log "Removing downloaded archive"
  rm -f "$TARBALL_NAME"

  log "Installing dependencies"
  sudo ./bin/installdependencies.sh

  log "Exiting directory for GitHub Actions Runner"
  popd
}

main() {
  # Software
  installPrerequisites

  # Memory
  setupZramSwap
  setupKernelVmParameters # Values chosen for ZRAM

  # Disks
  setupBtrfsMntFsVolume
  setupBtrfsMntFsSubvolumes

  # WARP for IPv6
  setupWarp

  # Nix
  setupNix
  # setupAttic

  # Network
  setupKernelNetParameters
  setupBpftune

  # GitHub Actions Runner
  setupGitHubActionsRunner
}

main
