{
  modulesPath,
  pkgs,
  ...
}: {
  imports = [
    "${modulesPath}/virtualisation/azure-common.nix"
    ./systemd/services/mount-nvmes.nix
    ./systemd/services/fix-tmp-permissions.nix
  ];
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

  environment = {
    memoryAllocator.provider = "mimalloc";
    # TODO(@connorbaker): Cargo-cult, or necessary for remote builders to work?
    # variables.NIX_REMOTE = "daemon";
  };

  networking = {
    hostName = "nixos-builder";
    nameservers = [
      "1.1.1.1"
      "8.8.8.8"
    ];
  };

  nix = {
    daemonCPUSchedPolicy = "batch";
    daemonIOSchedPriority = 7;
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
      extra-trusted-substituters = [
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
      trusted-users = ["root" "@nixbld" "@wheel"];
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
      # enable = true;
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
        autoSubUidGidRange = true;
        description = "Connor Baker's user account";
        extraGroups = ["wheel"];
        isNormalUser = true;
        openssh.authorizedKeys.keys = [
          "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJXpenPZWADrxK4+6nFmPspmYPPniI3m+3PxAfjbslg+ connorbaker@Connors-MacBook-Pro.local"
          "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJLd6kNEt/f89JGImBViXake15Y3VQ6AuKR/IBr1etpt connorbaker@nixos-desktop"
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
