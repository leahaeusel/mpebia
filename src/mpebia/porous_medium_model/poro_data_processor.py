"""Data processor class for the output of the porous medium model."""

import logging
from pathlib import Path

import numpy as np
import pyvista as pv
from fourcipp.fourc_input import FourCInput
from queens.data_processors._data_processor import (  # pylint: disable=import-error, no-name-in-module
    DataProcessor,
)

_logger = logging.getLogger(__name__)


class PoroDataProcessor(DataProcessor):
    """Class for extracting data from 4C output via Pyvista."""

    def __init__(
        self,
        file_name_identifier=None,
        file_options_dict=None,
    ):
        """Instantiate data processor class for the porous medium model.

        Args:
            file_name_identifier (str): Identifier of file name. The file prefix can contain regex
                expression and subdirectories.
            file_options_dict (dict): Dictionary with read-in options for the file
        """
        if "delete_field_data" not in file_options_dict:
            file_options_dict["delete_field_data"] = False

        super().__init__(
            file_name_identifier=file_name_identifier,
            file_options_dict=file_options_dict,
        )

        self.coords = self.get_coordinates(file_options_dict["input_file"])
        self.dline_coords = None

        if "dlines" in file_options_dict:
            if not len(file_options_dict["dlines"]) == len(file_options_dict["data_fields"]):
                raise ValueError("'data_fields' and 'dlines' must have the same length")

            self.dline_coords = self.get_coordinates_at_dline(
                file_options_dict["input_file"], file_options_dict["dlines"]
            )

    def get_raw_data_from_file(self, file_path):
        """Get the raw data from the files of interest.

        Args:
            file_path (str): Path to the file of interest.

        Returns:
            np.array: Raw data from file.
        """
        try:
            # not really raw data yet but only reader obj
            # we still need to specify which time steps to read
            raw_data = pv.get_reader(file_path)
            _logger.info("Successfully initiated reader object for %s.", file_path)
            return raw_data
        except FileNotFoundError as error:
            _logger.warning(
                "Could not find the file: %s. The following FileNotFoundError was raised: %s. "
                "Skipping the file and continuing.",
                file_path,
                error,
            )
        except ValueError as error:
            _logger.warning(
                "Could not read the file: %s. The following ValueError was raised: %s. "
                "Skipping the file and continuing.",
                file_path,
                error,
            )
        return None

    def filter_and_manipulate_raw_data(self, raw_data, coords_all_fields):
        """Filter and manipulate the raw data.

        Args:
            raw_data (np.array): Raw data from file.
            coords_all_fields (list, np.array): Coordinates for all or for each data field.

        Returns:
            np.array: Data of all the provided fields.
        """
        data_fields = self.file_options_dict["data_fields"]
        data_all_fields = []

        if not isinstance(coords_all_fields, list):
            coords_all_fields = [coords_all_fields] * len(data_fields)

        for data_field, coords in zip(data_fields, coords_all_fields):
            eval_points = pv.PolyData(coords)
            data = []
            for time_step in self.file_options_dict["time_steps"]:
                raw_data.set_active_time_point(time_step)
                mesh = raw_data.read()[0]
                sampled_data = eval_points.sample(mesh)
                data.append(np.array(sampled_data.point_data[data_field]))
            data = np.array(data).squeeze()
            data_all_fields.append(data)

        return data_all_fields

    def get_data_from_file(self, base_dir_file, at_dline=True):
        """Get data (at dline) from file.

        Args:
            base_dir_file (Path): Path of the base directory that contains the file of interest.
            at_dline (bool): Whether to extract data at dline coordinates or at all coordinates.

        Returns:
            np.array: Final data.
        """
        if not base_dir_file:
            raise ValueError(
                "The data processor requires a base_directory for the "
                "files to operate on! Your input was empty! Abort..."
            )
        if not isinstance(base_dir_file, Path):
            raise TypeError(
                "The argument 'base_dir_file' must be of type 'Path' "
                f"but is of type {type(base_dir_file)}. Abort..."
            )
        if at_dline and self.dline_coords is None:
            at_dline = False

        file_path = self._check_file_exist_and_is_unique(base_dir_file)
        data = None
        if file_path:
            raw_data = self.get_raw_data_from_file(file_path)
            if at_dline:
                data = self.filter_and_manipulate_raw_data(
                    raw_data, coords_all_fields=self.dline_coords
                )
            else:
                data = self.filter_and_manipulate_raw_data(raw_data, coords_all_fields=self.coords)

        return data

    def get_coordinates(self, input_file):
        """Get coordinates from 4C input file.

        Args:
            input_file (str, Path): Path to 4C input file.

        Returns:
            np.array: Array of coordinates.
        """
        fourc_input = FourCInput.from_4C_yaml(input_file)
        node_coords = fourc_input["NODE COORDS"]

        coords = []
        for node_coord in node_coords:
            coords.append(node_coord["COORD"])

        return np.array(coords)

    def get_coordinates_at_dline(self, input_file, dlines):
        """Extract coordinates at given DLINE from 4C input file.

        Args:
            input_file (str, Path): Path to 4C input file.
            dlines (list): List of DLINE ids to extract coordinates at.

        Returns:
            np.array: Array of coordinates at DLINE.
        """
        fourc_input = FourCInput.from_4C_yaml(input_file)
        topology = fourc_input["DLINE-NODE TOPOLOGY"]
        coords = fourc_input["NODE COORDS"]

        all_dline_coords = []
        for dline in dlines:
            dline_coords = []
            for node in topology:
                if node["d_id"] == dline:
                    node_id = node["node_id"]
                    coords_at_node = coords[node_id - 1]

                    if not coords_at_node["id"] == node_id:
                        raise RuntimeError(
                            "The node ids in the topology and the node coordinates are not ordered the "
                            "same. Abort..."
                        )

                    dline_coords.append(coords_at_node["COORD"])
                    continue
            all_dline_coords.append(np.array(dline_coords))

        return all_dline_coords
