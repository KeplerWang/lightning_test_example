import os

from lightning.pytorch.cli import SaveConfigCallback
from lightning.fabric.utilities.cloud_io import get_filesystem


class WandBSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer, pl_module, stage):
        ##### only changed the config_path
        assert trainer.log_dir is not None
        assert not self.save_to_log_dir, '{save_to_log_dir} must be False for WandBSaveConfigCallback'

        log_dir = os.path.join(trainer.log_dir, trainer.logger.name, trainer.logger.version)
        config_path = os.path.join(log_dir, self.config_filename)
        #####
        #### code below is same as SaveConfigCallback.setup
        fs = get_filesystem(log_dir)

        if not self.overwrite:
            # check if the file exists on rank 0
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            # broadcast whether to fail to all ranks
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                )

        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )
