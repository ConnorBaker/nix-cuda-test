from __future__ import annotations

from pydantic import BaseModel

from lambda_vm.components import schemas


class Unauthorized(schemas.ErrorResponseBody):
    """
    Unauthorized.
    """

    pass


class Forbidden(schemas.ErrorResponseBody):
    """
    Forbidden.
    """

    pass


class NotFound(schemas.ErrorResponseBody):
    """
    Object does not exist.
    """

    pass


class BadRequest(schemas.ErrorResponseBody):
    """
    Request parameters were invalid.
    """

    pass


class InternalServerError(schemas.ErrorResponseBody):
    """
    Something unexpected occurred.
    """

    pass


class Instances(BaseModel):
    """
    OK.
    """

    data: list[schemas.Instance]


class Instance(BaseModel):
    """
    OK.
    """

    data: schemas.Instance


class InstanceTypes(BaseModel):
    """
    OK.
    """

    data: dict[schemas.InstanceTypeName, DataInstanceType]

    class DataInstanceType(BaseModel):
        """
        Attributes:
            instance_type: Information about an instance type
            regions_with_capacity_available: List of regions, if any, that have this
                instance type available
        """

        instance_type: schemas.InstanceType
        regions_with_capacity_available: list[schemas.Region]


class Launch(BaseModel):
    """
    OK.
    """

    data: DataLaunch

    class DataLaunch(BaseModel):
        """
        Attributes:
            instance_ids: The unique identifiers (IDs) of the launched instances. Note:
                if a quantity was specified, fewer than the requested quantity might
                have been launched.
        """

        instance_ids: list[schemas.InstanceId]


class Terminate(BaseModel):
    """
    OK.
    """

    data: DataTerminate

    class DataTerminate(BaseModel):
        """
        Attributes:
            terminated_instances: List of instances that were terminated. Note: this
                list might not contain all instances requested to be terminated.
        """

        terminated_instances: list[schemas.Instance]


class Restart(BaseModel):
    """
    OK.
    """

    data: DataRestart

    class DataRestart(BaseModel):
        """
        Attributes:
            restarted_instances: List of instances that were restarted. Note: this list
                might not contain all instances requested to be restarted.
        """

        restarted_instances: list[schemas.Instance]


class SshKeys(BaseModel):
    """
    OK.

    Attributes:
        data: List of SSH public keys.
    """

    data: list[schemas.SshKey]


class AddSSHKey(BaseModel):
    """
    OK.

    Attributes:
        data: The added or generated SSH public key. If a new key pair was generated,
            the response body contains a `private_key` property that *must* be saved
            locally. Lambda Cloud does not store private keys.
    """

    data: schemas.SshKey


class FileSystems(BaseModel):
    """
    OK.

    Attributes:
        data: List of file systems.
    """

    data: list[schemas.FileSystem]
