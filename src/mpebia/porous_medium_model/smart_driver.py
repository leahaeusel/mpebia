"""Driver to recycle the model evaluations from a previous QUEENS run."""

import logging
import os

from queens.drivers import Jobscript
from queens.utils.exceptions import SubprocessError
from queens.utils.logger_settings import log_init_args
from queens.utils.run_subprocess import run_subprocess

from mpebia.porous_medium_model.recycle_metadata import RecycleMetadata, get_metadata_from_job_dir

_logger = logging.getLogger(__name__)


class SmartDriver(Jobscript):
    """Driver to recycle the model evaluations from a previous QUEENS run.

    Attributes:
        input_template (Path): Read in simulation input template as string
        data_processor (obj): Instance of data processor class
        gradient_data_processor (obj): Instance of data processor class for gradient data
        jobscript_template (str): Read in jobscript template as string
        jobscript_options (dict): Dictionary containing jobscript options
        jobscript_file_name (str): Jobscript file name (default: 'jobscript.sh')
        raise_error_on_jobscript_failure (bool): Whether to raise an error for a non-zero jobscript
                                                 exit code. If False, a warning is logged instead.
        reuse_existing_jobs (bool): Whether to reuse existing jobs if the output files already exist
                                    and the input parameters are the same.
    """

    @log_init_args
    def __init__(
        self,
        parameters,
        input_templates,
        jobscript_template,
        executable,
        files_to_copy=None,
        data_processor=None,
        gradient_data_processor=None,
        jobscript_file_name="jobscript.sh",
        extra_options=None,
        raise_error_on_jobscript_failure=True,
        reuse_existing_jobs=True,
    ):
        """Initialize SmartDriver object.

        Args:
            parameters (Parameters): Parameters object.
            input_templates (str, Path, dict): Path(s) to simulation input template.
            jobscript_template (str, Path): Path to jobscript template or read-in jobscript
                                            template.
            executable (str, Path): Path to main executable of respective software.
            files_to_copy (list, opt): Files or directories to copy to experiment_dir.
            data_processor (obj, opt): Instance of data processor class.
            gradient_data_processor (obj, opt): Instance of data processor class for gradient data.
            jobscript_file_name (str, opt): Jobscript file name (default: 'jobscript.sh').
            extra_options (dict, opt): Extra options to inject into jobscript template.
            raise_error_on_jobscript_failure (bool, opt): Whether to raise an error for a non-zero
                                                          jobscript exit code.
            reuse_existing_jobs (bool, opt): Whether to reuse existing jobs if the output files
                                             already exist and the input parameters are the same.
        """
        super().__init__(
            parameters=parameters,
            input_templates=input_templates,
            jobscript_template=jobscript_template,
            executable=executable,
            files_to_copy=files_to_copy,
            data_processor=data_processor,
            gradient_data_processor=gradient_data_processor,
            jobscript_file_name=jobscript_file_name,
            extra_options=extra_options,
            raise_error_on_jobscript_failure=raise_error_on_jobscript_failure,
        )
        self.reuse_existing_jobs = reuse_existing_jobs

    def run(self, sample, job_id, num_procs, experiment_dir, experiment_name):
        """Run the driver.

        Args:
            sample (dict): Sample for which to run the driver
            job_id (int): Job ID
            num_procs (int): Number of processors
            experiment_dir (Path): Path to QUEENS experiment directory
            experiment_name (str): Name of QUEENS experiment

        Returns:
            Result and potentially the gradient
        """
        job_dir = experiment_dir / str(job_id)

        try:
            original_metadata = get_metadata_from_job_dir(job_dir)
            if original_metadata is None:
                original_metadata = {}
        except FileNotFoundError:
            original_metadata = {}

        jobscript_status = (
            original_metadata.get("times", {}).get("run_jobscript", {}).get("status", "")
        )

        if self.reuse_existing_jobs and jobscript_status == "successful":
            # Make sure input parameters are the same
            inputs = self.parameters.sample_as_dict(sample)

            if not original_metadata["inputs"] == inputs:
                raise ValueError(
                    f"Input parameters of the original QUEENS run and the SmartDriver run must be "
                    f"equal for job ID {job_id} when reusing existing output files."
                )

            new_metadata_filename = self._new_metadata_filename(job_dir)
            metadata = RecycleMetadata(
                job_id=job_id,
                inputs=inputs,
                job_dir=job_dir,
                metadata_filename=new_metadata_filename,
            )

            with metadata.time_code("data_processing"):
                output_dir = job_dir / "output"
                results = self._get_results(output_dir)
                metadata.outputs = results
        else:
            results = super().run(sample, job_id, num_procs, experiment_dir, experiment_name)

        return results

    def _run_executable(self, job_id, execute_cmd):
        """Run executable.

        Args:
            job_id (int): Job ID.
            execute_cmd (str): Executed command.
        """
        process_returncode, _, stdout, stderr = run_subprocess(
            execute_cmd,
            raise_error_on_subprocess_failure=False,
        )
        if self.raise_error_on_jobscript_failure and process_returncode:
            raise SubprocessError.construct_error_from_command(
                command=execute_cmd,
                command_output=stdout,
                error_message=stderr,
                additional_message=f"The jobscript with job ID {job_id} has failed with exit code "
                f"{process_returncode}.",
            )
        elif process_returncode:
            _logger.warning(
                f"The jobscript with job ID {job_id} has failed with exit code "
                f"{process_returncode}."
            )

    def _new_metadata_filename(self, job_dir):
        """Find the next metadata file name.

        This is to avoid overwriting of already existing metadata files

        Returns:
            str: New metadata filename
        """
        recycle_run = 1
        metadata_filename = self.metadata_filename(recycle_run)

        while os.path.isfile(job_dir / (metadata_filename + ".yaml")):
            recycle_run += 1
            metadata_filename = self.metadata_filename(recycle_run)

        return metadata_filename

    @staticmethod
    def metadata_filename(recycle_run):
        """Metadata filename for a specific run of the recycle driver.

        Args:
            recycle_run (int): Counter of the number of recycle runs

        Returns:
            str: metadata filename of the recycle run
        """
        return f"metadata_recycle_run_{recycle_run}"
