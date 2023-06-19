import logging

import requests

from lambda_vm.components import request_bodies, responses, schemas

URL = "https://cloud.lambdalabs.com/api/v1/"


def instance_types(api_key: str) -> responses.InstanceTypes:
    """
    Returns a detailed list of the instance types offered by Lambda GPU
    Cloud. The details include the regions, if any, in which each instance
    type is currently available.

    Args:
        api_key: Your API key.
    """
    logging.info("Getting instance types")
    response = requests.get(
        f"{URL}/instance-types",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    return responses.InstanceTypes(**response.json())


def list_instances(api_key: str) -> responses.Instances:
    """
    Retrieves a detailed list of running instances.

    Args:
        api_key: Your API key.
    """
    logging.info("Getting running instances")
    response = requests.get(
        f"{URL}/instances",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    return responses.Instances(**response.json())


def get_instance(api_key: str, id: schemas.InstanceId) -> responses.Instance:
    """
    Retrieves details of a specific instance, including whether or not the
    instance is running.

    Args:
        api_key: Your API key.
        id: The unique identifier (ID) of the instance.
    """
    logging.info(f"Getting details for instance {id}")
    response = requests.get(
        f"{URL}/instances/{id}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    return responses.Instance(**response.json())


def launch_instance(api_key: str, launch: request_bodies.Launch) -> responses.Launch:
    """
    Launches one or more instances of a given instance type.

    Args:
        api_key: Your API key.
        launch: The request body for the launch endpoint.
    """
    logging.info(f"Launching {launch.quantity} instance(s)")
    response = requests.post(
        f"{URL}/instance-operations/launch",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=launch.model_dump_json(),
    )
    response.raise_for_status()
    return responses.Launch(**response.json())


def terminate_instance(api_key: str, terminate: request_bodies.Terminate) -> responses.Terminate:
    """
    Terminates a given instance.
    """
    logging.info(f"Terminating instances {terminate.instance_ids}")
    response = requests.post(
        f"{URL}/instance-operations/terminate",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=terminate.model_dump_json(),
    )
    response.raise_for_status()
    return responses.Terminate(**response.json())


def restart_instance(api_key: str, restart: request_bodies.Restart) -> responses.Restart:
    """
    Restarts the given instances.
    """
    logging.info(f"Restarting instances {restart.instance_ids}")
    response = requests.post(
        f"{URL}/instance-operations/restart",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=restart.model_dump_json(),
    )
    response.raise_for_status()
    return responses.Restart(**response.json())


def list_ssh_keys(api_key: str) -> responses.SshKeys:
    """
    Retrieve the list of SSH keys

    Args:
        api_key: Your API key.
    """
    logging.info("Getting SSH keys")
    response = requests.get(f"{URL}/ssh-keys", headers={"Authorization": f"Bearer {api_key}"})
    response.raise_for_status()
    return responses.SshKeys(**response.json())


def add_ssh_key(api_key: str, add_ssh_key: request_bodies.AddSSHKey) -> responses.AddSSHKey:
    """
    Add an SSH key

    To use an existing key pair, enter the public key for the `public_key` property of
    the request body.

    To generate a new key pair, omit the `public_key` property from the request body.
    Save the `private_key` from the response somewhere secure. For example, with curl:

    ```
    curl https://cloud.lambdalabs.com/api/v1/ssh-keys \
        --fail \
        -u ${LAMBDA_API_KEY}: \
        -X POST \
        -d '{"name": "new key"}' \
        | jq -r '.data.private_key' > key.pem

    chmod 400 key.pem
    ```

    Then, after you launch an instance with `new key` attached to it:
    ```
    ssh -i key.pem <instance IP>
    ```
    """
    logging.info(f"Adding SSH key {add_ssh_key.name}")
    response = requests.post(
        f"{URL}/ssh-keys",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=add_ssh_key.model_dump_json(),
    )
    response.raise_for_status()
    return responses.AddSSHKey(**response.json())


def delete_ssh_key(api_key: str, ssh_key_id: schemas.SshKeyId) -> None:
    """
    Delete an SSH key.

    Args:
        api_key: Your API key.
        id: The unique identifier (ID) of the SSH key.
    """
    logging.info(f"Deleting SSH key {id}")
    response = requests.delete(
        f"{URL}/ssh-keys/{id}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    response.raise_for_status()
    return


def list_file_systems(api_key: str) -> responses.FileSystems:
    """
    Retrieve the list of file systems

    Args:
        api_key: Your API key.
    """
    logging.info("Getting file systems")
    response = requests.get(f"{URL}/file-systems", headers={"Authorization": f"Bearer {api_key}"})
    response.raise_for_status()
    return responses.FileSystems(**response.json())
