from chap_core.adaptors.command_line_interface import generate_app

from chtorch.estimator import Estimator, ProblemConfiguration, ModelConfiguration

app = generate_app(Estimator(ProblemConfiguration(prediction_length=3, debug=True), ModelConfiguration(context_length=12)))
app()
