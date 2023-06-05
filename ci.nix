# Used by Hercules CI
let
  # Note: This is incredibly gross, and I am sorry for all who stumble upon this.
  inherit (builtins.getFlake (toString ./.)) inputs;
  inherit (inputs.nixpkgs-lib) lib;
  inherit (lib) attrsets flip lists pipe strings versions;
  system = builtins.currentSystem;

  # Read the redistributables to get a sense of which CUDA versions are supported.
  # redistributableCudaVersions :: List Version
  redistributableCudaVersions = pipe "${inputs.nixpkgs.outPath}/pkgs/development/compilers/cudatoolkit/redist/manifests" [
    builtins.readDir
    builtins.attrNames
    (builtins.map (strings.removePrefix "redistrib_"))
    (builtins.map (strings.removeSuffix ".json"))
    # TODO(@connorbaker): PyTorch doesn't currently support 12.0+
    (lists.filter (flip strings.versionOlder "12.0.0"))
  ];

  # dotsToFlats :: String -> String
  dotsToFlats = builtins.replaceStrings ["."] ["_"];

  # The name of the package set for a given CUDA version.
  # mkCudaPackagesName :: Version -> String
  mkCudaPackagesName = flip pipe [
    versions.majorMinor
    dotsToFlats
    (version: "cudaPackages_${version}")
  ];

  # supportedCapabilities :: Version -> List Capability
  supportedCapabilities = cudaVersion: let
    # isSupported :: Gpu -> Bool
    isSupported = gpu: let
      inherit (gpu) minCudaVersion maxCudaVersion;
      lowerBoundSatisfied = strings.versionAtLeast cudaVersion minCudaVersion;
      upperBoundSatisfied =
        (maxCudaVersion == null)
        || !(strings.versionOlder maxCudaVersion cudaVersion);
    in
      lowerBoundSatisfied && upperBoundSatisfied;

    # gpus :: List Gpu
    gpus = builtins.import "${inputs.nixpkgs.outPath}/pkgs/development/compilers/cudatoolkit/gpus.nix";

    # GPUs which are supported by the provided CUDA version.
    # supportedGpus :: List Gpu
    supportedGpus = pipe gpus [
      (lists.filter isSupported)
      # Jetson is a whole different beast.
      (lists.filter (gpu: gpu.computeCapability != "8.7"))
    ];
  in
    builtins.map (gpu: gpu.computeCapability) supportedGpus;

  # combinations :: List (Attr Set)
  combinations = builtins.concatMap (cudaVersion:
    attrsets.cartesianProductOfSets {
      cudaCapabilities = builtins.map lists.singleton (supportedCapabilities cudaVersion);
      cudaPackages = [(mkCudaPackagesName cudaVersion)];
      cudaForwardCompat = [false];
    })
  redistributableCudaVersions;
  # For each version of CUDA, and for each supported cuda capaiblity, we
  # generate a package set.
  flakeBuilder = config @ {
    cudaCapabilities,
    cudaPackages,
    cudaForwardCompat,
  }:
    inputs.flake-parts.lib.mkFlake {inherit inputs;} {
      systems = [system];
      imports = [./nix];
      perSystem = _: {
        inherit config;
      };
    };
in
  builtins.listToAttrs (builtins.map (config: {
      name = with config; "${cudaPackages}-cc-${
        dotsToFlats (builtins.head cudaCapabilities)
      }-ptx-${
        if cudaForwardCompat
        then "yes"
        else "no"
      }";
      value = attrsets.recurseIntoAttrs (flakeBuilder config).packages.${system};
    })
    combinations)
