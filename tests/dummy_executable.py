"""Dummy executable."""

import sys
from pathlib import Path

import numpy as np
from numpy import genfromtxt


def main(params):
    """Dummy main function.

    Args:
        params (dict): Dictionary with input parameters.

    Returns:
        np.array: Dummy function output
    """
    c = np.linspace(0, 1, 4)
    return params["x1"] * np.sin(c)


def write_results(output, output_path):
    """Write solution to csv files."""
    y = output
    output_file = output_path.parent / (output_path.stem + "_output.csv")
    if y.shape[0] != 0:
        np.savetxt(output_file, y, delimiter=",")


def read_input_file(input_file_path):
    """Read-in input from csv file."""
    inputs = genfromtxt(input_file_path, delimiter=r",|\s+")
    return inputs


if __name__ == "__main__":
    parameters = read_input_file(input_file_path=sys.argv[1])
    print(parameters)
    main_output = main(params={"x1": parameters})
    print(main_output)
    write_results(main_output, output_path=Path(sys.argv[2]))
