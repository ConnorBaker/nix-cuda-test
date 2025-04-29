final: prev: {
  pythonPackagesExtensions = prev.pythonPackagesExtensions or [ ] ++ [
    (finalPythonPackages: prevPythonPackages: {
      nix-cuda-sr = finalPythonPackages.callPackage ./nix-cuda-sr { };
    })
  ];
}
