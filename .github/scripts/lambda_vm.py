import requests
import argparse
import logging
import sys
from time import sleep
from dataclasses import dataclass
from typing import Literal, get_args, TypedDict

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
)

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


class Specs(TypedDict):
    vcpus: int
    memory_gib: int
    storage_gib: int


class Region(TypedDict):
    name: str
    description: str


class InstanceType(TypedDict):
    name: InstanceTypeName
    description: str
    price_cents_per_hour: int
    specs: Specs


API_URL = "https://cloud.lambdalabs.com/api/v1"


class DataInstanceType(TypedDict):
    instance_type: InstanceType
    regions_with_capacity_available: list[Region]


class InstanceTypes(TypedDict):
    data: dict[InstanceTypeName, DataInstanceType]


class Instance(TypedDict):
    id: str
    status: Literal["active", "booting", "unhealthy", "terminated"]
    ssh_key_names: list[str]
    file_system_names: list[str]
    name: None | str
    ip: None | str
    region: None | Region
    instance_type: None | InstanceType
    hostname: None | str
    jupyter_token: None | str
    jupyter_url: None | str


class Instances(TypedDict):
    data: list[Instance]


class LaunchRequestBody(TypedDict):
    instance_type_name: InstanceTypeName
    ssh_key_names: list[str]  # Must be a single key
    region_name: str


class InstanceIds(TypedDict):
    instance_ids: list[str]


class LaunchResponse(TypedDict):
    data: InstanceIds


class Details(TypedDict):
    data: Instance


@dataclass
class Config:
    api_key: str
    instance_type_name: InstanceTypeName


def get_instance_types(api_key: str) -> InstanceTypes:
    logging.info("Getting instance types")
    response = requests.get(
        f"{API_URL}/instance-types",
        headers={"Authorization": f"Basic {api_key}"},
    )
    response.raise_for_status()
    j: InstanceTypes = response.json()
    logging.info(f"Got instance types: {j}")
    return j


def get_instances(api_key: str) -> Instances:
    logging.info("Getting running instances")
    response = requests.get(
        f"{API_URL}/instances",
        headers={"Authorization": f"Basic {api_key}"},
    )
    response.raise_for_status()
    j: Instances = response.json()
    logging.info(f"Got running instances: {j}")
    return j


def check_if_already_running(config: Config) -> None:
    # Get running instances
    instances = get_instances(config.api_key)
    # Filter for matching type
    matching: list[Instance] = []
    for instance in instances["data"]:
        instance_type = instance["instance_type"]
        assert instance_type is not None
        if instance_type["name"] == config.instance_type_name:
            matching.append(instance)

    if len(matching) == 1:
        logging.warning("Found running instance, exiting...")
        sys.exit(0)
    elif len(matching) > 1:
        logging.error("Found multiple running instances, exiting with error...")
        sys.exit(1)


def check_if_available(config: Config) -> DataInstanceType:
    # Get available instance types
    instance_types = get_instance_types(config.api_key)
    available: list[DataInstanceType] = []
    matching: list[DataInstanceType] = []

    for data_instance_type in instance_types["data"].values():
        if data_instance_type["regions_with_capacity_available"] != []:
            available.append(data_instance_type)
            if data_instance_type["instance_type"]["name"] == config.instance_type_name:
                matching.append(data_instance_type)

    if len(matching) == 0:
        logging.error(
            f"No available instances found for {config.instance_type_name}."
            f" Consider using one of the available instances: {available}."
            " Exiting with error..."
        )
        sys.exit(1)
    elif len(matching) > 1:
        logging.error(
            "Found multiple entries for the same instance type, which should not"
            " happen. Exiting with error..."
        )
        sys.exit(1)

    return matching[0]


def launch_instance(api_key: str, body: LaunchRequestBody) -> LaunchResponse:
    logging.info("Launching instance")
    response = requests.post(
        f"{API_URL}/instance-operations/launch",
        headers={
            "Authorization": f"Basic {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
    )
    response.raise_for_status()
    j = response.json()
    logging.info(f"Launched instance: {j}")
    return j


def instance_details(api_key: str, instance_id: str) -> Details:
    logging.info(f"Getting details for instance {instance_id}")
    response = requests.get(
        f"{API_URL}/instances/{instance_id}",
        headers={"Authorization": f"Basic {api_key}"},
    )
    response.raise_for_status()
    j: Details = response.json()
    logging.info(f"Got details for instance {instance_id}: {j}")
    return j


# If successful in creating an instance, returns its IP address
def main(config: Config) -> str:
    # Check if already running
    check_if_already_running(config)
    logging.info("No running instances found, creating one...")

    # Check if available
    data_instance_type = check_if_available(config)
    logging.info("Found availability for requested instance type")

    launch_request_body = LaunchRequestBody(
        instance_type_name=data_instance_type["instance_type"]["name"],
        region_name=data_instance_type["regions_with_capacity_available"][0]["name"],
        ssh_key_names=["github-runner"],
    )
    logging.info(f"Launch request body: {launch_request_body}")

    # Launch instance
    launched = launch_instance(
        config.api_key,
        launch_request_body,
    )
    id = launched["data"]["instance_ids"][0]

    # Wait for instance to be active
    logging.info(
        f"Waiting for instance {id} to be active, querying status every 30s..."
    )
    details = instance_details(config.api_key, id)
    while details["data"]["status"] != "active":
        if details["data"]["status"] == "booting":
            logging.info(f"Instance {id} is booting, waiting...")
            sleep(30)
        elif details["data"]["status"] == "unhealthy":
            logging.error("Instance is unhealthy, exiting with error...")
            sys.exit(1)
        elif details["data"]["status"] == "terminated":
            logging.error("Instance terminated, exiting with error...")
            sys.exit(1)

        details = instance_details(config.api_key, id)

    ip = details["data"]["ip"]
    if ip is None:
        logging.error(
            "Instance is active, but has no IP address, exiting with error..."
        )
        sys.exit(1)
    else:
        logging.info(f"Instance {id} is active at IP address {ip}.")
        return ip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance-type-name",
        choices=get_args(InstanceTypeName),
        required=True,
        help="Name of instance type to launch",
    )
    parser.add_argument("--key", type=str, required=True, help="API key")
    args = parser.parse_args()
    config = Config(api_key=args.key, instance_type_name=args.instance_type_name)
    ip = main(config)
    print(ip)
