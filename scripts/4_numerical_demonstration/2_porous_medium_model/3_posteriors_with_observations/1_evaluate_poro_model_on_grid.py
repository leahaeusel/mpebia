"""Evaluate poro model on a grid of different Young's moduli."""

import logging
from pathlib import Path

from queens.distributions import FreeVariable
from queens.global_settings import GlobalSettings
from queens.main import run_iterator
from queens.models import Simulation
from queens.parameters.parameters import Parameters
from queens.utils.remote_operations import RemoteConnection

from mpebia.output import get_directory
from mpebia.porous_medium_model.parameters_shared import ParametersShared as params
from mpebia.porous_medium_model.poro_data_processor import PoroDataProcessor
from mpebia.porous_medium_model.poro_grid_iterator import PoroGridIterator
from mpebia.porous_medium_model.poro_schedulers import PoroClusterScheduler
from mpebia.porous_medium_model.smart_driver import SmartDriver
from mpebia.porous_medium_model.truncated_normal_distribution import TruncatedNormal

logger = logging.getLogger(__name__)


################################################################
# Needs to be adapted personally:
fourc_build_dir = "<path_to_remote_4C_executable>"
input_dir_cluster = "<path_to_remote_poro_geometries>"
cluster_ip = "<ip_address_remote_cluster>"
cluster_user = "<username_remote_cluster>"
remote_python = "<path_to_remote_env_manager>/envs/mpebia/bin/python"
remote_queens_repository = Path("<path_to_remote_queens_repository>")
local_queens_repository = Path("<path_to_local_queens_repository>")
################################################################

directory = get_directory(__file__)
input_dir = (directory / "..").resolve()

fourc_inputs = ["poro_input_with_blood.4C.yaml", "poro_input_without_blood.4C.yaml"]
fourc_geometries = ["poro_geometry_with_arteries.4C.yaml", "poro_geometry_without_arteries.4C.yaml"]
experiment_names = ["mpebia_poro_grid_with_blood", "mpebia_poro_grid_without_blood"]
data_field_lst = [
    ["ale-displacement", "volfrac_blood_lung"],
    ["ale-displacement"],
]
dlines_lst = [[8, 9], [8]]

for input, geometry, experiment_name, data_fields, dlines in zip(
    fourc_inputs, fourc_geometries, experiment_names, data_field_lst, dlines_lst
):
    fourc_input_path = input_dir / input
    fourc_geometry = input_dir / geometry
    global_settings = GlobalSettings(experiment_name=experiment_name, output_dir=directory)

    with global_settings as gs:
        # Parameters
        young_h = TruncatedNormal(
            mean=params.mean_prior[0],
            std=params.std_prior[0],
            lower_bound=params.truncations_Eh[0],
            upper_bound=params.truncations_Eh[1],
        )
        young_d = TruncatedNormal(
            mean=params.mean_prior[1],
            std=params.std_prior[1],
            lower_bound=params.truncations_Ed[0],
            upper_bound=params.truncations_Ed[1],
        )
        k_j = FreeVariable(dimension=1)
        input_dir_var = FreeVariable(dimension=1)
        parameters = Parameters(young_h=young_h, young_d=young_d, k_j=k_j, input_dir=input_dir_var)

        data_processor = PoroDataProcessor(
            file_name_identifier="*porofluid.pvd",
            file_options_dict={
                "time_steps": [-1],
                "data_fields": data_fields,
                "input_file": fourc_geometry,
                "dlines": dlines,
            },
        )

        # Cluster stuff
        remote_connection = RemoteConnection(
            host=cluster_ip,
            user=cluster_user,
            remote_python=remote_python,
            remote_queens_repository=remote_queens_repository,
        )
        scheduler = PoroClusterScheduler(
            workload_manager="slurm",
            walltime="24:00:00",
            queue="normal",
            num_jobs=10,
            num_procs=8,
            num_nodes=1,
            remote_connection=remote_connection,
            experiment_name=gs.experiment_name,
        )
        driver = SmartDriver(
            parameters=parameters,
            input_templates=fourc_input_path,
            jobscript_template=local_queens_repository / "templates/jobscripts/fourc_thought.sh",
            executable=fourc_build_dir + "/4C",
            data_processor=data_processor,
            extra_options={
                "cluster_script": "/lnm/share/donottouch.sh",
            },
            raise_error_on_jobscript_failure=False,
            reuse_existing_jobs=True,
        )

        forward_model = Simulation(scheduler=scheduler, driver=driver)

        iterator = PoroGridIterator(
            grid_design={
                "young_h": {
                    "num_grid_points": params.num_grid_points_Eh,
                    "axis_type": "ppf",
                    "offset_ppf": params.offset_ppf,
                    "data_type": "FLOAT",
                },
                "young_d": {
                    "num_grid_points": params.num_grid_points_Ed,
                    "axis_type": "ppf",
                    "offset_ppf": params.offset_ppf,
                    "data_type": "FLOAT",
                },
                "k_j": {"axis_type": "fix", "value": params.k_j, "num_grid_points": 1},
                "input_dir": {"axis_type": "fix", "value": input_dir_cluster, "num_grid_points": 1},
            },
            result_description={
                "write_results": True,
            },
            model=forward_model,
            parameters=parameters,
            global_settings=gs,
        )

        # Actual analysis
        run_iterator(iterator, global_settings=gs)
