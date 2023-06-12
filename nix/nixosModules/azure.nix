{modulesPath, ...}: {
  imports = ["${modulesPath}/virtualisation/azure-common.nix"];
  system.stateVersion = "23.05";

  boot = {
    initrd = {
      compressor = "zstd";
      compressorArgs = ["-19"];
      kernelModules = ["nvme"];
      # TODO: Check out services.swraid.mdadmConf;
    };
    kernelParams = ["nvme_core.io_timeout=4294967295"];
  };

  environment.memoryAllocator.provider = "mimalloc";

  fileSystems = let
    defaults = {
      autoFormat = true;
      fsType = "ext4";
      options = [
        "defaults"
        "nofail"
        "X-mount.mkdir"
        "x-systemd.device-timeout=2min"
        "x-systemd.mount-timeout=2min"
      ];
    };
  in {
    "/nix/store" =
      {
        device = "/dev/nvme0n1";
      }
      // defaults;

    "/nested/disk1" =
      {
        device = "/dev/nvme1n1";
      }
      // defaults;
  };

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
      max-substitution-jobs = 256;
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
