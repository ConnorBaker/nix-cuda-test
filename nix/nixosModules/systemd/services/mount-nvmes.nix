{pkgs, ...}: {
  systemd.services.mount-nvmes = {
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
}
