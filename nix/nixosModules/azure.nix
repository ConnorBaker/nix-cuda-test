{pkgs, ...}: {
  system.stateVersion = "23.05";

  boot.initrd.availableKernelModules = ["nvme"];

  networking = {
    hostName = "nixos-builder";
    nameservers = [
      "1.1.1.1"
      "8.8.8.8"
    ];
  };

  services = {
    hercules-ci-agent = {
      enable = true;
      # TODO(@connorbaker): Need to evaluate whether we will run out of space using the default
      #   work directory in /var/lib.
      settings.concurrentTasks = 4;
    };
    openssh = {
      allowSFTP = false;
      settings.PasswordAuthentication = false;
    };
  };

  nix = {
    # Use a newer version of Nix to take advantage of max-substitution-jobs
    package = let
      inherit (pkgs) nix;
      inherit (pkgs.nixVersions) nix_2_16;
      inherit (pkgs.lib.strings) versionAtLeast;
    in
      if versionAtLeast nix.version "2.16"
      then nix
      else nix_2_16;

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
      max-substitution-jobs = 1024;
      narinfo-cache-negative-ttl = 0;
      system-features = [
        "benchmark"
        "big-parallel"
        "kvm"
        "nixos-test"
      ];
    };
  };

  zramSwap = {
    algorithm = "zstd";
    enable = true;
    memoryPercent = 200;
    # writebackDevice = "";
  };

  fileSystems = let
    options = [
      "defaults"
      "noatime"
      "noauto"
      "user"
    ];
  in {
    "~connorbaker" = {
      inherit options;
      device = "/dev/nvme0n1";
    };

    "/mnt/disk2" = {
      inherit options;
      device = "/dev/nvme1n1";
    };
  };

  security.sudo = {
    execWheelOnly = true;
    wheelNeedsPassword = false;
  };

  users = {
    mutableUsers = false;
    users = {
      connorbaker = {
        description = "Connor Baker's user account";
        extraGroups = ["wheel"];
        isNormalUser = true;
        openssh.authorizedKeys.keys = [
          "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCu3/mVyRa8kMAw1NWJAILmPvwigdEdNWozaG4Hh30vOu5BfCL+bCjWHtB2ohyUxCK4Z0/+rkIF2j4XOYndelKRuGXBnDkbXdLBCzUtFeDEMVPm2HcUHIPfa7pBIgaJAD01sMwWA7RwaYNFxiQ3yj2te6HQW8TuzE9XXZ6pn5ImUj6apykdiXfzNzZAe/HN3oVjRtPV2E//m+STs3fBOWeLGrQ2r72W2jxJKJN9+NDtZ5snsPpKd/LW457uqCPD0WbUEpeDdo7qMUO1GReF0F2psiPlDrDXH09fDMq7Nh3eCcTuCNvGoHAsogyMvD+vufGxETKdBWkL2m/1if/PllA3F7qdCc6lKsNpX0+HMAGSaiJUVHRRiTWOc79Q6Z9tfQAzrwH/wubi5xgcMIDzide/cKUPzBryCJrH5TiF4lvTynYLwvjii0hA7rhla8dK8yPdTbAt6GZROGQhT97UipMUY2fEP3o0f9GT6LgeGpwSfKXQtnT1/1LwbYTdAGRNZ78= connorbaker@Connors-MBP"
        ];
        packages = with pkgs; [
          curl
          gh
          git
          htop
          nixpkgs-review
          vim
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
}
