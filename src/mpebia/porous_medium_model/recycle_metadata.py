"""QUEENS metadata for recycle driver."""

from pathlib import Path

import yaml
from queens.utils.config_directories import job_dirs_in_experiment_dir
from queens.utils.metadata import SimulationMetadata

DEFAULT_METADATA_FILENAME = "metadata"
METADATA_FILETYPE = ".yaml"


class RecycleMetadata(SimulationMetadata):
    """Simulation metadata object for use with RecycleDriver.

    This objects holds metadata, times code sections and exports them to yaml.

    Attributes:
        job_id (int): Id of the job
        inputs (dict): Parameters for this job
        file_path (pathlib.Path): Path to export the metadata
        timestamp (str): Timestamp of the object creation
        outputs (tuple): Results obtain by the simulation
        times (dict): Wall times of code sections
    """

    def __init__(self, job_id, inputs, job_dir, metadata_filename=DEFAULT_METADATA_FILENAME):
        """Init simulation metadata object.

        Args:
            job_id (int): Id of the job
            inputs (dict): Parameters for this job
            job_dir (pathlib.Path): Directory in which to write the metadata
            metadata_filename (str, opt): Name of the metadata file
        """
        super().__init__(job_id, inputs, job_dir)
        self.file_path = (Path(job_dir) / metadata_filename).with_suffix(METADATA_FILETYPE)


def get_metadata_from_job_dir(job_dir, metadata_filename=DEFAULT_METADATA_FILENAME):
    """Get metadata from a job directory.

    Args:
        job_dir (pathlib.Path): Job directory
        metadata_filename (str, opt): Name of the metadata file

    Returns:
        metadata (dict): metadata of a job
    """
    metadata_path = (job_dir / metadata_filename).with_suffix(METADATA_FILETYPE)
    metadata = yaml.safe_load(metadata_path.read_text())
    return metadata


def get_metadata_from_experiment_dir(experiment_dir, metadata_filename=DEFAULT_METADATA_FILENAME):
    """Get metadata from experiment_dir.

    To keep memory usage limited, this is implemented as a generator.

    Args:
        experiment_dir (pathlib.Path, str): Path with the job dirs
        metadata_filename (str, opt): Name of the metadata file

    Yields:
        metadata (dict): metadata of a job
    """
    for job_dir in job_dirs_in_experiment_dir(experiment_dir):
        yield get_metadata_from_job_dir(job_dir, metadata_filename)
