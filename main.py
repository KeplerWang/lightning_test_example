from lightning.pytorch.cli import LightningCLI, ArgsType

from utils import WandBSaveConfigCallback


def cli_main(args: ArgsType = None):
    cli = LightningCLI(
        seed_everything_default=42,
        save_config_callback=WandBSaveConfigCallback,
        save_config_kwargs={'save_to_log_dir': False},
        args=args,
    )


if __name__ == '__main__':
    cli_main()
