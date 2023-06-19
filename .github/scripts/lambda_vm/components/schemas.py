from typing import List, Literal, NewType

from pydantic import BaseModel

"""
Unique identifier for the type of error
"""
ErrorCode = Literal[
    "global/unknown",
    "global/invalid-api-key",
    "global/account-inactive",
    "global/invalid-parameters",
    "global/object-does-not-exist",
    "instance-operations/launch/insufficient-capacity",
    "instance-operations/launch/file-system-in-wrong-region",
    "instance-operations/launch/file-systems-not-supported",
    "ssh-keys/key-in-use",
]


class Error(BaseModel):
    """
    Error object returned by the API

    Attributes:
        code: Unique identifier for the type of error
        message: Detailed description of the error
        suggestion: Suggestion of possible ways to fix the error
    """

    code: ErrorCode
    message: str
    suggestion: None | str = None


class ErrorResponseBody(BaseModel):
    """
    Error response body returned by the API

    Attributes:
        error: Error object returned by the API
        field_errors: Details about errors on a per-parameter basis
    """

    error: Error
    field_errors: None | dict[str, Error] = None


"""
A date and time, formatted as an ISO 8601 time stamp
"""
DateTime = NewType("DateTime", str)


UserStatus = Literal["active", "deactivated"]


class User(BaseModel):
    """
    Information about a user in your team

    Attributes:
        id: Unique identifier for the user
        email: Email address of the user
        status: Status of the user
    """

    id: str
    email: str
    status: UserStatus


"""
Short name of a region
"""
RegionName = NewType("RegionName", str)


class Region(BaseModel):
    """
    Information about a region

    Attributes:
        name: Short name of a region
        description: Long name of a region
    """

    name: RegionName
    description: str


"""
Unique identifier (ID) of an instance
"""
InstanceId = NewType("InstanceId", str)

"""
Unique identifier (ID) of an SSH key
"""
SshKeyId = NewType("SshKeyId", str)

"""
Unique identifier (ID) of a file system
"""
FileSystemId = NewType("FileSystemId", str)

"""
Name of the SSH key
"""
SshKeyName = NewType("SshKeyName", str)

"""
Public key for the SSH key
"""
SshPublicKey = NewType("SshPublicKey", str)

"""
Private key for the SSH key. Only returned when generating a new key pair.
"""
SshPrivateKey = None | NewType("SshPrivateKey", str)


class SshKey(BaseModel):
    """
    Information about a stored SSH key, which can be used to access instances over SSH

    Attributes:
        id: Unique identifier (ID) of an SSH key
        name: Name of the SSH key
        public_key: Public key for the SSH key
        private_key: Private key for the SSH key. Only returned when generating a new
            key pair.
    """

    id: SshKeyId
    name: SshKeyName
    public_key: SshPublicKey
    private_key: SshPrivateKey = None


"""
Name of a file system
"""
FileSystemName = NewType("FileSystemName", str)


class FileSystem(BaseModel):
    """
    Information about a shared file system

    Attributes:
        id: Unique identifier (ID) of a file system
        name: Name of a file system
        created: Date and time when the file system was created
        created_by: User who created the file system
        mount_point: Absolute path indicating where on instances the file system will be
            mounted
        region: Region where the file system is located
        is_in_use: Whether the file system is currently in use by an instance. File
            systems that are in use cannot be deleted.
    """

    id: FileSystemId
    name: FileSystemName
    created: DateTime
    created_by: User
    mount_point: str
    region: Region
    is_in_use: bool


"""
Name of an instance type
"""
InstanceTypeName = Literal[
    "gpu_1x_a10",
    "gpu_1x_a100",
    "gpu_1x_a100_sxm4",
    "gpu_1x_a6000",
    "gpu_1x_h100_pcie",
    "gpu_1x_rtx6000",
    "gpu_2x_a100",
    "gpu_2x_a6000",
    "gpu_4x_a100",
    "gpu_4x_a6000",
    "gpu_8x_a100",
    "gpu_8x_a100_80gb_sxm4",
    "gpu_8x_v100",
]

"""
User-provided name for the instance
"""
InstanceName = None | NewType("InstanceName", str)


class Specs(BaseModel):
    """
    Instance type specifications

    Attributes:
        vcpus: Number of virtual CPUs
        memory_gib: Amount of memory in GiB
        storage_gib: Amount of storage in GiB
    """

    vcpus: int
    memory_gib: int
    storage_gib: int


class InstanceType(BaseModel):
    """
    Hardware configuration and pricing of an instance type

    Attributes:
        name: Name of an instance type
        description: Long name of the instance type
        price_cents_per_hour: Price per hour in cents
        specs: Instance type specifications
    """

    name: InstanceTypeName
    description: str
    price_cents_per_hour: int
    specs: Specs


InstanceStatus = Literal["active", "booting", "unhealthy", "terminated"]


class Instance(BaseModel):
    """
    Virtual machine (VM) in Lambda Cloud

    Attributes:
        id: Unique identifier (ID) of an instance
        status: Status of the instance
        ssh_key_names: Names of the SSH keys allowed to access the instance
        file_system_names: Names of the file systems, if any, attached to the instance
        region: Region where the instance is located
        instance_type: Instance type
        name: User-provided name for the instance
        ip: IP address of the instance
        hostname: Hostname of the instance
        jupyter_token: Token for accessing JupyterLab on the instance
        jupyter_url: URL for accessing JupyterLab on the instance
    """

    id: InstanceId
    status: InstanceStatus
    ssh_key_names: List[SshKeyName]
    file_system_names: List[FileSystemName]
    region: Region
    instance_type: InstanceType
    name: InstanceName = None
    ip: None | str = None
    hostname: None | str = None
    jupyter_token: None | str = None
    jupyter_url: None | str = None
