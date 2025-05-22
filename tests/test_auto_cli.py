from chap_core.adaptors.command_line_interface import generate_template_app

from chtorch.model_template import ExposedModelTemplate


def test_model_template_info():
    info = ExposedModelTemplate().model_template_info
    print(info)


def test_write_yaml():
    info = ExposedModelTemplate()
    app, *_, write_yaml = generate_template_app(info)
    y = write_yaml()
    print(y)
