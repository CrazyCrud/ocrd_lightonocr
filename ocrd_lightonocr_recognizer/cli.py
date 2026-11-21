import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from .recognize import OcrdLightOnOcrRecognizer


@click.command()
@ocrd_cli_options
def ocrd_lightonocr_recognizer(*args, **kwargs):
    """
    CLI entrypoint for ocrd-lightonocr-recognizer.
    """

    return ocrd_cli_wrap_processor(OcrdLightOnOcrRecognizer, *args, **kwargs)


if __name__ == "__main__":
    ocrd_lightonocr_recognizer()