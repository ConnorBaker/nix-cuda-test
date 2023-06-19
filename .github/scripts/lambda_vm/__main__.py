import json
import logging
from argparse import ArgumentParser, Namespace
from sys import exit
from time import sleep
from typing import Literal, get_args

from pydantic import BaseModel

from lambda_vm import paths
from lambda_vm.components import request_bodies, responses, schemas

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
)

Action = Literal["start", "terminate"]


class CliArgs(BaseModel):
    action: Action
    instance_type_name: schemas.InstanceTypeName
    api_key: str


def check_if_already_running(cli_args: CliArgs) -> None | schemas.Instance:
    # Get running instances
    instances: responses.Instances = paths.list_instances(cli_args.api_key)
    # Filter for matching type
    matching: list[schemas.Instance] = []
    for instance in instances.data:
        name: schemas.InstanceTypeName = instance.instance_type.name
        if name == cli_args.instance_type_name:
            matching.append(instance)

    match len(matching):
        case 0:
            return None
        case 1:
            return matching[0]
        case _:
            logging.error("Found multiple running instances, exiting with error...")
            exit(1)


def check_if_available(cli_args: CliArgs) -> responses.InstanceTypes.DataInstanceType:
    # Get available instance types
    instance_types: responses.InstanceTypes = paths.instance_types(cli_args.api_key)
    available: list[responses.InstanceTypes.DataInstanceType] = []
    matching: list[responses.InstanceTypes.DataInstanceType] = []

    for instance_type_name, data_instance_type in instance_types.data.items():
        if data_instance_type.regions_with_capacity_available != []:
            available.append(data_instance_type)
            if instance_type_name == cli_args.instance_type_name:
                matching.append(data_instance_type)

    if len(matching) == 0:
        logging.error(f"No available instances found for {cli_args.instance_type_name}.")
        logging.error(
            "Consider using one of the available instances: "
            + json.dumps(
                {
                    instance.instance_type.name: ", ".join(
                        region.name for region in instance.regions_with_capacity_available
                    )
                    for instance in available
                },
                indent=4,
                sort_keys=True,
            )
        )
        logging.error("Exiting with error...")
        exit(1)
    elif len(matching) > 1:
        logging.error("Found multiple entries for the same instance type, which should not" " happen.")
        logging.error("Exiting with error...")
        exit(1)

    return matching[0]


def wait_for_status(api_key: str, status: schemas.InstanceStatus, id: schemas.InstanceId) -> responses.Instance:
    # Wait for instance to be active
    logging.info(f"Waiting for instance {id} to be active, will query status every 30s.")
    details = paths.get_instance(api_key, id)
    actual_status: schemas.InstanceStatus = details.data.status
    while actual_status != status:
        # Special cases for when we're stuck in an undesired state
        if actual_status == "unhealthy":
            logging.error("Instance is unhealthy, exiting with error...")
            exit(1)

        logging.info(f"Instance {id} is {actual_status}, waiting for transition to {status}...")
        sleep(30)
        return wait_for_status(api_key, status, id)

    logging.info(f"Instance {id} has desired status {status}, continuing...")
    return details


def start_instance(cli_args: CliArgs) -> str:
    # Check if available
    data_instance_type: responses.InstanceTypes.DataInstanceType = check_if_available(cli_args)
    logging.info("Found availability for requested instance type")

    launch_request_body: request_bodies.Launch = request_bodies.Launch(
        instance_type_name=data_instance_type.instance_type.name,
        name=f"github-runner-{cli_args.instance_type_name}",
        region_name=data_instance_type.regions_with_capacity_available[0].name,
        ssh_key_names=[schemas.SshKeyName("github-runner")],
    )
    logging.info(f"Launch request body: {launch_request_body.model_dump_json()}")

    # Launch instance
    launched: responses.Launch = paths.launch_instance(
        cli_args.api_key,
        launch_request_body,
    )
    launched_instance_ids: list[schemas.InstanceId] = launched.data.instance_ids
    assert len(launched_instance_ids) == 1
    id: schemas.InstanceId = launched_instance_ids[0]
    details: responses.Instance = wait_for_status(cli_args.api_key, "active", id)

    ip: None | str = details.data.ip
    if ip is None:
        logging.error("Instance is active, but has no IP address, exiting with error...")
        exit(1)
    else:
        logging.info(f"Instance {id} is active at IP address {ip}.")
        return ip


def terminate_instance(cli_args: CliArgs, instance: schemas.Instance) -> str:
    # Terminate instance
    terminated: responses.Terminate = paths.terminate_instance(
        cli_args.api_key,
        request_bodies.Terminate(instance_ids=[instance.id]),
    )
    logging.info(f"Sent request to terminate instance(s) {terminated.data.model_dump_json()}.")
    terminated_instances: list[schemas.Instance] = terminated.data.terminated_instances
    assert len(terminated_instances) == 1
    terminated_instance = terminated_instances[0]
    id: schemas.InstanceId = terminated_instance.id
    wait_for_status(cli_args.api_key, "terminated", id)
    logging.info(f"Instance {id} is terminated.")
    ip: None | str = terminated_instance.ip
    if ip is None:
        logging.error("Instance is terminated, but had no IP address, exiting with error...")
        exit(1)
    else:
        logging.info(f"Instance {id} is terminated at IP address {ip}.")
    return ip


# If successful in creating an instance, returns its IP address
def main(cli_args: CliArgs) -> str:
    # Check if already running
    maybe_instance: None | schemas.Instance = check_if_already_running(cli_args)
    match (maybe_instance, cli_args.action):
        case (None, "start"):
            logging.info("No running instance found, creating one...")
            return start_instance(cli_args)
        case (_instance, "start"):
            logging.info("Instance already running, exiting...")
            exit(0)
        case (None, "terminate"):
            logging.info("No running instance found, exiting...")
            exit(0)
        case (instance, "terminate"):
            logging.info("Instance found, terminating...")
            return terminate_instance(cli_args, instance)

    logging.error("Unknown error, exiting with error...")
    exit(1)


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "--action",
        choices=get_args(Action),
        required=True,
        help="Action to perform",
    )
    parser.add_argument(
        "--instance-type-name",
        choices=get_args(schemas.InstanceTypeName),
        required=True,
        help="Name of instance type to launch",
    )
    parser.add_argument("--api-key", type=str, required=True, help="API key")
    args: Namespace = parser.parse_args()
    cli_args: CliArgs = CliArgs(
        action=args.action,
        api_key=args.api_key,
        instance_type_name=args.instance_type_name,
    )
    ip: str = main(cli_args)
    print(ip)
