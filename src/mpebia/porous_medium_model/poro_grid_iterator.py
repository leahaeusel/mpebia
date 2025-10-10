"""Driver to recycle the model evaluations from a previous QUEENS run."""

import logging
from importlib.metadata import distribution

import numpy as np
from queens.iterators import Grid
from queens.utils.process_outputs import process_outputs, write_results

_logger = logging.getLogger(__name__)


class PoroGridIterator(Grid):
    """Grid Iterator to evaluate the poro model posteriors.

    Enables evaluation of the poro model on a grid that is evenly spaced
    with respect to the cumulative distribution function of the prior.
    """

    def pre_run(self):
        """Generate samples based on description in *grid_dict*."""
        # pre-allocate empty list for filling up with vectors of grid points as elements
        grid_point_list = []

        #  set up 1D arrays for each parameter (needs bounds and type of axis)
        for parameter_name, parameter in self.parameters.dict.items():
            axis_type = self.grid_dict[parameter_name].get("axis_type", None)
            num_grid_points = self.grid_dict[parameter_name].get("num_grid_points", None)
            offset_ppf = self.grid_dict[parameter_name].get("offset_ppf", None)
            self.num_grid_points_per_axis.append(num_grid_points)
            self.scale_type.append(axis_type)

            if axis_type == "lin":
                start_value = parameter.lower_bound
                stop_value = parameter.upper_bound
                grid_point_list.append(
                    np.linspace(
                        start_value,
                        stop_value,
                        num=num_grid_points,
                        endpoint=True,
                        retstep=False,
                    )
                )
            elif axis_type == "ppf":
                # This is where the magic happens
                equally_spaced_ppf = np.linspace(offset_ppf, 1 - offset_ppf, num_grid_points)
                grid_point_list.append(parameter.ppf(equally_spaced_ppf))
            elif axis_type == "fix":
                grid_point_list.append(self.grid_dict[parameter_name].get("value", None))
            else:
                raise NotImplementedError(
                    "Invalid option for 'axis_type'. Valid options are: "
                    f"'lin', 'log10', 'ln'. You chose {axis_type}."
                )

        grid_coords = np.meshgrid(*grid_point_list)
        self.samples = np.empty(
            [np.prod(self.num_grid_points_per_axis), self.num_parameters], dtype=object
        )
        for i in range(self.num_parameters):
            self.samples[:, i] = grid_coords[i].flatten()

    def post_run(self):
        """Analyze the results."""
        if self.result_description is not None:
            results = {}
            results["raw_output_data"] = self.output
            results["input_data"] = self.samples
            if self.result_description["write_results"]:
                write_results(results, self.global_settings.result_file(".pickle"))
