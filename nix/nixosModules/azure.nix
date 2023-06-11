{
  modulesPath,
  pkgs,
  ...
}: {
  imports = ["${modulesPath}/virtualisation/azure-common.nix"];
  system.stateVersion = "23.05";

  boot = {
    initrd = {
      availableKernelModules = ["nvme"];
      compressor = "zstd";
      compressorArgs = ["-19"];
      kernelModules = ["nvme"];
      # TODO: Check out services.swraid.mdadmConf;
    };
    kernelParams = ["nvme_core.io_timeout=4294967295"];
  };

  environment.memoryAllocator.provider = "mimalloc";

  # fileSystems = let
  #   defaults = {
  #     autoFormat = true;
  #     fsType = "ext4";
  #     options = [
  #       "defaults"
  #       "noatime"
  #       "noauto"
  #       "user"
  #       "X-mount.mkdir"
  #     ];
  #   };
  # in {
  #   "/mnt/disk0" =
  #     {
  #       device = "/dev/nvme0n1";
  #     }
  #     // defaults;

  #   "/mnt/disk1" =
  #     {
  #       device = "/dev/nvme1n1";
  #     }
  #     // defaults;
  # };

  networking = {
    hostName = "nixos-builder";
    nameservers = [
      "1.1.1.1"
      "8.8.8.8"
    ];
  };

  nix = {
    settings = {
      accept-flake-config = true;
      allow-import-from-derivation = false;
      cores = 0;
      experimental-features = [
        "flakes"
        "nix-command"
        "no-url-literals"
      ];
      extra-substituters = [
        "https://cantcache.me"
        "https://cuda-maintainers.cachix.org"
      ];
      extra-trusted-public-keys = [
        "cantcache.me:Y+FHAKfx7S0pBkBMKpNMQtGKpILAfhmqUSnr5oNwNMs="
        "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
      ];
      extra-trusted-users = ["@nixbld" "@wheel" "connorbaker" "runner"];
      fsync-metadata = false;
      http-connections = 0;
      keep-derivations = true;
      keep-outputs = true;
      max-jobs = "auto";
      max-substitution-jobs = 256;
      narinfo-cache-negative-ttl = 0;
      system-features = [
        "benchmark"
        "big-parallel"
        "kvm"
        "nixos-test"
      ];
    };
  };

  nixpkgs = {
    config = {
      allowUnfree = true;
      cudaCapabilities = ["8.6"];
      cudaSupport = true;
    };
    overlays = [
      # Need newer version of Nix supporting max-substitution-jobs
      (_: prev: {
        nix = prev.nixVersions.nix_2_16;
      })
      # Must be disabled to use mimalloc
      (_: prev: {
        dhcpcd = prev.dhcpcd.override {enablePrivSep = false;};
      })
    ];
  };

  security.sudo = {
    execWheelOnly = true;
    wheelNeedsPassword = false;
  };

  services = {
    hercules-ci-agent = {
      enable = true;
      # TODO(@connorbaker): Need to evaluate whether we will run out of space using the default
      #   work directory in /var/lib.
      settings.concurrentTasks = 256;
    };
    openssh = {
      allowSFTP = false;
      enable = true;
      settings.PasswordAuthentication = false;
    };
  };

  systemd.services = let
    mount-nvme-script = device: mountPoint: ''
      #!/usr/bin/env bash
      set -euo pipefail

      # Check if the drive is already mounted
      if mountpoint -q ${mountPoint}; then
        echo "Drive already mounted, exiting..."
        exit 0
      fi
      echo "Drive not mounted, continuing..."

      # Check if the drive is already formatted
      if blkid ${device}; then
        echo "Drive already formatted, continuing..."
      else
        echo "Drive not formatted, formatting..."
        mkfs.ext4 ${device}
      fi

      # Check if the mount point exists
      if [ -d ${mountPoint} ]; then
        echo "Mount point exists, continuing..."
      else
        echo "Mount point does not exist, creating..."
        mkdir -p ${mountPoint}
      fi

      # Mount the drive
      echo "Mounting drive..."
      mount ${device} ${mountPoint}
    '';
  in {
    mount-nvme0n1 = {
      description = "Mount the first NVMe drive";
      path = with pkgs; [e2fsprogs util-linux];
      script = mount-nvme-script "/dev/nvme0n1" "/mnt/disk0";
      wantedBy = ["multi-user.target"];
    };
    mount-nvme1n1 = {
      description = "Mount the second NVMe drive";
      path = with pkgs; [e2fsprogs util-linux];
      script = mount-nvme-script "/dev/nvme1n1" "/mnt/disk1";
      wantedBy = ["multi-user.target"];
    };
  };

  users = {
    mutableUsers = false;
    users = {
      connorbaker = {
        description = "Connor Baker's user account";
        extraGroups = ["wheel"];
        isNormalUser = true;
        openssh.authorizedKeys.keys = [
          "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJXpenPZWADrxK4+6nFmPspmYPPniI3m+3PxAfjbslg+ connorbaker@Connors-MacBook-Pro.local"
        ];
      };
      runner = {
        description = "GitHub runner user account";
        extraGroups = ["wheel"];
        isNormalUser = true;
        openssh.authorizedKeys.keys = [
          "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBgxYZuzBSHhETkrNcilkzsLBTGoHhXSa9ug6KwHkNDz github-runner"
        ];
      };
    };
  };

  zramSwap = {
    algorithm = "zstd";
    enable = true;
    memoryPercent = 200;
  };
}
