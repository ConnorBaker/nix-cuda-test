from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from lambda_vm.components import schemas


class Launch(BaseModel):
    """
    Request body for the launch endpoint

    Attributes:
        region_name: Name of the region to launch the instance in
        instance_type_name: Name of the instance type to launch
        ssh_key_names: Names of the SSH keys to add to the instance. Must be a single
            key.
        file_system_names: Names of the file systems to attach to the instance
        quantity: Number of instances to launch
    """
    region_name: schemas.RegionName
    instance_type_name: schemas.InstanceTypeName
    ssh_key_names: list[schemas.SshKeyName]
    file_system_names: list[schemas.FileSystemName] = []
    quantity: Literal[1] = 1
    name: schemas.InstanceName = None


class Terminate(BaseModel):
    """
    Request body for the terminate endpoint

    Attributes:
        instance_ids: The unique identifiers (IDs) of the instances to terminate
    """
    instance_ids: list[schemas.InstanceId]


class Restart(BaseModel):
    """
    Request body for the restart endpoint

    Attributes:
        instance_ids: The unique identifiers (IDs) of the instances to restart
    """
    instance_ids: list[schemas.InstanceId]


class AddSSHKey(BaseModel):
    """
    The name for the SSH key. Optionally, an existing public key can be supplied for the
    `public_key` property. If the `public_key` property is omitted, a new key pair is
    generated. The private key is returned in the response.

    Attributes:
        
    """
    name: schemas.SshKeyName
    public_key: None | schemas.SshPublicKey = None