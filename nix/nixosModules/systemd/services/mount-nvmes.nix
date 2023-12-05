{pkgs, ...}:
let
  app = pkgs.writeShellApplication {
    name = "mount-nvme-drives";
    runtimeInputs = with pkgs; [
      e2fsprogs
      lvm2
      util-linux
    ];
    text = ''
      # Wait for any nvme*n1 devices to appear in 10 second intervals
      while ! ls /dev/nvme*n1; do
        sleep 10
      done

      # Create the physical volumes
      for dev in /dev/nvme*n1; do
        if ! pvcreate "$dev"; then
          echo "Failed to create physical volume on $dev"
          exit 1
        fi
      done

      # Create the volume group
      if ! vgcreate ext_vg /dev/nvme*n1; then
        echo "Failed to create volume group ext_vg"
        exit 1
      fi

      # Create the logical volume
      if ! lvcreate -l "100%FREE" -n ext_lv ext_vg; then
        echo "Failed to create logical volume ext_lv"
        exit 1
      fi

      # Format the logical volume
      if ! mkfs.ext4 -L nixos-store /dev/ext_vg/ext_lv; then
        echo "Failed to format logical volume ext_lv"
        exit 1
      fi

      # Create and mount the mount points with overlayfs
      mkdir -p /ext
      if ! mount /dev/ext_vg/ext_lv /ext; then
        echo "Failed to mount logical volume ext_lv at /ext"
        exit 1
      fi

      # Create the union filesystems
      for dir in home nix tmp var; do
        mkdir -p "/ext/$dir/upperdir" "/ext/$dir/workdir"
        if ! mount -t overlay -o lowerdir=/"$dir",upperdir=/ext/"$dir"/upperdir,workdir=/ext/"$dir"/workdir none /"$dir"; then
          echo "Failed to mount overlayfs at /$dir"
          exit 1
        fi
      done

      # Fix permissions on /tmp
      if ! chmod 1777 /tmp; then
        echo "Failed to set permissions on /tmp"
        exit 1
      fi

      # Restart nix-daemon.service and nix-daemon.socket
      if ! systemctl restart nix-daemon.service nix-daemon.socket; then
        echo "Failed to restart nix-daemon.service and nix-daemon.socket"
        exit 1
      fi
    '';
  };
in
{
  systemd.services.mount-nvmes = {
    description = "Mount the NVMe drives";
    script = app.text;
    serviceConfig = {
      RemainAfterExit = "yes";
      Type = "oneshot";
    };
    wantedBy = ["multi-user.target"];
  };
}
