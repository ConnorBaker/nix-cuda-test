{
  modulesPath,
  pkgs,
  ...
}: {
  imports = ["${modulesPath}/virtualisation/azure-common.nix"];
  system.stateVersion = "23.05";

  boot = {
    initrd = {
      compressor = "zstd";
      compressorArgs = ["-19"];
      kernelModules = ["nvme"];
    };
    kernelPackages = pkgs.linuxPackages_latest;
    tmp.cleanOnBoot = true;
  };

  environment.memoryAllocator.provider = "mimalloc";

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
      fsync-metadata = false;
      http-connections = 0;
      keep-derivations = true;
      keep-outputs = true;
      max-jobs = "auto";
      max-substitution-jobs = 1024;
      narinfo-cache-negative-ttl = 0;
      system-features = [
        "benchmark"
        "big-parallel"
        "kvm"
        "nixos-test"
      ];
      trusted-users = ["root" "@nixbld" "@wheel" "connorbaker" "runner"];
    };
  };

  nixpkgs = {
    config = {
      allowUnfree = true;
      # Use the default cudaCapabilities
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

  systemd.services = {
    mount-nvmes = {
      description = "Mount the NVMe drives";
      path = with pkgs; [e2fsprogs lvm2 util-linux];
      script =
      # Set up the service
      ''
        #!/usr/bin/env bash
        set -euo pipefail
      ''
      # Wait for the devices to be available
      + ''
        while [[ ! -e /dev/nvme0n1 || ! -e /dev/nvme1n1 ]]; do
            sleep 1
        done
      ''
      # Create the physical volumes
      + ''
        pvcreate /dev/nvme0n1
        pvcreate /dev/nvme1n1
      ''
      # Create the volume group
      + ''
        vgcreate ext_vg /dev/nvme0n1 /dev/nvme1n1
      ''
      # Create the logical volume
      + ''
        lvcreate -l "100%FREE" -n ext_lv ext_vg
      ''
      # Format the logical volume
      + ''
        mkfs.ext4 -L nixos-store /dev/ext_vg/ext_lv
      ''
      # Create and mount the mount points with overlayfs
      + ''
        mkdir -p /ext
        mount /dev/ext_vg/ext_lv /ext
        for dir in home nix tmp var; do
          mkdir -p /ext/$dir/{upper,work}dir
          mount \
            -t overlay \
            -o lowerdir=/$dir,upperdir=/ext/$dir/upperdir,workdir=/ext/$dir/workdir \
            none \
            /$dir
        done
      '';
      serviceConfig = {
        RemainAfterExit = "yes";
        Type = "oneshot";
      };
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
