"""Evaluate the porous medium model with the true Young's moduli."""

import pickle

import numpy as np
from queens.distributions import FreeVariable
from queens.global_settings import GlobalSettings
from queens.models import Simulation
from queens.parameters.parameters import Parameters
from queens_interfaces.fourc.driver import Fourc

from mpebia.output import get_directory
from mpebia.porous_medium_model.parameters_shared import ParametersShared
from mpebia.porous_medium_model.poro_data_processor import PoroDataProcessor
from mpebia.porous_medium_model.poro_schedulers import PoroLocalScheduler
from mpebia.porous_medium_model.recycle_driver import RecycleDriver

if __name__ == "__main__":

    ################################################################
    # Needs to be adapted personally:
    fourc_build_dir = "<path_to_local_4C_executable>"
    ################################################################

    directory = get_directory(__file__)
    input_dir = (directory / "..").resolve()
    fourc_input_template = input_dir / "poro_input_with_blood.4C.yaml"
    fourc_geometry = input_dir / "poro_geometry_with_arteries.4C.yaml"

    experiment_name = "mpebia_poro_ground_truth"

    global_settings = GlobalSettings(experiment_name=experiment_name, output_dir=directory)

    with global_settings as gs:
        young_h = FreeVariable(dimension=1)
        young_d = FreeVariable(dimension=1)
        k_j = FreeVariable(dimension=1)
        input_dir_var = FreeVariable(dimension=1)
        parameters = Parameters(young_h=young_h, young_d=young_d, k_j=k_j, input_dir=input_dir_var)

        data_processors = [
            PoroDataProcessor(
                file_name_identifier="*porofluid.pvd",
                file_options_dict={
                    "time_steps": [-1],
                    "data_fields": ["ale-displacement", "volfrac_blood_lung"],
                    "input_file": fourc_geometry,
                    "dlines": [8, 9],
                },
            ),
            PoroDataProcessor(
                file_name_identifier="*porofluid.pvd",
                file_options_dict={
                    "time_steps": [-1],
                    "data_fields": ["ale-displacement", "volfrac_blood_lung"],
                    "input_file": fourc_geometry,
                },
            ),
        ]
        suffixes = ["_at_dline", "_full"]
        for i, (data_processor, suffix) in enumerate(zip(data_processors, suffixes)):
            scheduler = PoroLocalScheduler(
                experiment_name=gs.experiment_name,
                num_procs=6,
            )

            if i == 0:
                driver = Fourc(
                    parameters=parameters,
                    input_templates=fourc_input_template,
                    executable=fourc_build_dir + "/4C",
                    data_processor=data_processor,
                )
            else:
                driver = RecycleDriver(
                    parameters=parameters,
                    data_processor=data_processor,
                )

            forward_model = Simulation(scheduler=scheduler, driver=driver)

            # Actual analysis
            input_parameters = np.array(
                [
                    [
                        ParametersShared.young_h_gt,
                        ParametersShared.young_d_gt,
                        ParametersShared.k_j,
                        str(input_dir),
                    ]
                ]
            )
            output = forward_model.evaluate(input_parameters)

            if i == 0:
                coords = data_processor.dline_coords
            else:
                coords = data_processor.coords

            with open(gs.result_file(".pickle", suffix), "wb") as f:
                pickle.dump(
                    {"input_parameters": input_parameters, "output": output, "coords": coords}, f
                )
