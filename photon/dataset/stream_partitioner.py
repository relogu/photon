"""Partition the streams in a YAML file into multiple clients, IID."""

import sys
from logging import INFO
from pathlib import Path

import yaml
from flwr.common.logger import log


def partition_streams(input_file: Path, output_file: Path, num_clients: int) -> None:
    """Partition the streams in a YAML file into multiple clients, IID."""
    # Load the original YAML file

    data = [
        value
        for _i, current_client_stream in enumerate(yaml.safe_load(input_file.open()))
        for _k, (_, streams) in enumerate(current_client_stream.items())
        for _j, (_key, value) in enumerate(streams.items())
    ]

    assert (
        len(data) % num_clients == 0
    ), f"Number of streams must be divisible by num_clients, {len(data)}, {num_clients}"

    # Partition the data into num_clients lists
    step = len(data) // num_clients
    partitioned_data = [data[i * step : (i + 1) * step] for i in range(num_clients)]

    # Transform back to the original format, i.e. a list of dictionaries
    output_data = [
        {
            "client_streams": {
                f"stream_{i}": stream
                for i, stream in enumerate(partitioned_data[client])
            },
        }
        for client in range(num_clients)
    ]

    output_file.write_text(yaml.dump(output_data))


if __name__ == "__main__":
    NUM_ARGS = 4
    if len(sys.argv) != NUM_ARGS:
        log(
            INFO,
            """Usage: python stream_partitioner.py
            <input_file> <output_file> <num_clients>""",
        )
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    num_clients = int(sys.argv[3])

    partition_streams(input_file, output_file, num_clients)
