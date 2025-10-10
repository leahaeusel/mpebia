"""Integration test for the smart driver."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from queens.data_processors import CsvFile
from queens.distributions import Uniform
from queens.drivers import Jobscript
from queens.global_settings import GlobalSettings
from queens.iterators import Grid
from queens.main import run_iterator
from queens.models import Simulation
from queens.parameters import Parameters
from queens.schedulers import Local
from queens.utils.config_directories import experiment_directory
from queens.utils.io import load_result

from mpebia.porous_medium_model.smart_driver import SmartDriver


def test_smart_driver_with_previous_run(
    tmp_path,
    executable,
    input_file,
    jobscript_template,
    parameters,
    data_processor,
    _create_input_file_for_executable,
):
    """Integration test for the smart driver."""
    global_settings = GlobalSettings(
        experiment_name="test_smart_driver_with_previous_run", output_dir=tmp_path
    )

    results_jobscript_driver_run = run_jobscript_driver(
        executable,
        input_file,
        jobscript_template,
        global_settings,
        parameters,
        data_processor,
    )

    with global_settings:
        smart_driver = SmartDriver(
            parameters=parameters,
            input_templates=input_file,
            executable=executable,
            data_processor=data_processor,
            jobscript_template=jobscript_template,
            reuse_existing_jobs=True,
        )
        results_smart_driver = run_iterator_with_driver(smart_driver, global_settings, parameters)

    np.testing.assert_array_equal(
        results_smart_driver["input_data"],
        results_jobscript_driver_run["input_data"],
    )

    np.testing.assert_array_equal(
        results_smart_driver["raw_output_data"]["result"],
        results_jobscript_driver_run["raw_output_data"]["result"],
    )

    # Check metadata files
    experiment_dir = experiment_directory(experiment_name=global_settings.experiment_name)

    metadata_file = experiment_dir / "0" / "metadata.yaml"
    assert metadata_file.exists(), f"Metadata file {metadata_file} does not exist"

    metadata_file = experiment_dir / "0" / "metadata_recycle_run_1.yaml"
    assert metadata_file.exists(), f"Metadata file {metadata_file} does not exist"

    delete_simulation_data(global_settings)


def test_smart_driver_without_previous_run(
    tmp_path,
    executable,
    input_file,
    jobscript_template,
    parameters,
    data_processor,
    _create_input_file_for_executable,
):
    """Integration test for the smart driver."""
    global_settings = GlobalSettings(
        experiment_name="test_smart_driver_without_previous_run", output_dir=tmp_path
    )

    with global_settings:
        smart_driver = SmartDriver(
            parameters=parameters,
            input_templates=input_file,
            executable=executable,
            data_processor=data_processor,
            jobscript_template=jobscript_template,
            reuse_existing_jobs=True,
        )
        results_smart_driver = run_iterator_with_driver(smart_driver, global_settings, parameters)

    np.testing.assert_array_equal(
        results_smart_driver["input_data"],
        np.array([[-2], [2]]),
    )

    # Check metadata files
    experiment_dir = experiment_directory(experiment_name=global_settings.experiment_name)

    metadata_file = experiment_dir / "0" / "metadata.yaml"
    assert metadata_file.exists(), f"Metadata file {metadata_file} does not exist"

    metadata_file = experiment_dir / "0" / "metadata_recycle_run_1.yaml"
    assert not metadata_file.exists(), f"Metadata file {metadata_file} exists"

    delete_simulation_data(global_settings)


def run_jobscript_driver(
    executable,
    input_file,
    jobscript_template,
    global_settings,
    parameters,
    data_processor,
):
    """Run dummy executable with a jobscript driver and return the results.

    Args:
        executable (str, Path): Executable
        input_file (str, Path): Input file
        jobscript_template (str, Path): Jobscript template
        global_settings (GlobalSettings): Global settings
        parameters (Parameters): Parameters
        data_processor (DataProcessor): Data processor

    Returns:
        dict: Results
    """
    with global_settings:
        driver = Jobscript(
            parameters=parameters,
            input_templates=input_file,
            executable=executable,
            data_processor=data_processor,
            jobscript_template=jobscript_template,
        )
        results = run_iterator_with_driver(driver, global_settings, parameters)
    return results


def run_iterator_with_driver(driver, global_settings, parameters):
    """Run a QUEENS iterator with a given driver and return the results."""
    scheduler = Local(
        num_procs=1,
        num_jobs=1,
        experiment_name=global_settings.experiment_name,
    )
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = Grid(
        grid_design={
            "x1": {"num_grid_points": 2, "axis_type": "lin", "data_type": "FLOAT"},
        },
        result_description={
            "write_results": True,
        },
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    run_iterator(iterator, global_settings=global_settings)
    results = load_result(global_settings.result_file(".pickle"))

    return results


def delete_simulation_data(global_settings):
    """Delete all simulation data in the experiment directory."""
    experiment_dir = experiment_directory(experiment_name=global_settings.experiment_name)
    shutil.rmtree(str(experiment_dir))


@pytest.fixture(name="executable")
def fixture_executable():
    """Path to executable."""
    dir_of_this_file = Path(os.path.abspath(__file__)).parent
    executable = dir_of_this_file / "dummy_executable.py"
    return executable


@pytest.fixture(name="input_file")
def fixture_input_file(tmp_path):
    """Path to input file."""
    input_file = tmp_path / "input_file.csv"
    return input_file


@pytest.fixture(name="_create_input_file_for_executable")
def fixture_create_input_file_for_executable(input_file):
    """Create a csv file as temporary input file for executable."""
    input_file.write_text("{{ x1 }}", encoding="utf-8")


@pytest.fixture(name="jobscript_template")
def fixture_jobscript_template():
    """Jobscript template."""
    jobscript_template = "python {{ executable }} {{ input_file }} {{ output_file }}"
    return jobscript_template


@pytest.fixture(name="parameters")
def fixture_parameters():
    """Create QUEENS parameters."""
    x1 = Uniform(lower_bound=-2.0, upper_bound=2.0)
    parameters = Parameters(x1=x1)
    return parameters


@pytest.fixture(name="data_processor")
def fixture_data_processor():
    """Create QUEENS data processor."""
    data_processor = CsvFile(
        file_name_identifier="*_output.csv",
        file_options_dict={"delete_field_data": False, "filter": {"type": "entire_file"}},
    )
    return data_processor
