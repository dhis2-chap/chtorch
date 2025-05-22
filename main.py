from chtorch.model_template import ExposedModelTemplate

from chap_core.adaptors.command_line_interface import generate_template_app

app, *_ = generate_template_app(ExposedModelTemplate())
if __name__ == '__main__':
    app()

