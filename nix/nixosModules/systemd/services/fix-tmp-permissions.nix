{
  systemd.services.fix-tmp-permissions = {
    description = "Fixes permissions for /tmp";
    script =
      # Set up the service
      ''
        #!/usr/bin/env bash
        set -euo pipefail
      ''
      # Fix the permissions
      + ''
        chmod 1777 /tmp
      '';
    serviceConfig = {
      RemainAfterExit = "yes";
      Type = "oneshot";
    };
    wantedBy = ["multi-user.target"];
  };
}
