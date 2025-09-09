=======
CHTorch
=======


.. image:: https://img.shields.io/pypi/v/chtorch.svg
        :target: https://pypi.python.org/pypi/chtorch

.. image:: https://img.shields.io/travis/knutdrand/chtorch.svg
        :target: https://travis-ci.com/knutdrand/chtorch

.. image:: https://readthedocs.org/projects/chtorch/badge/?version=latest
        :target: https://chtorch.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Utility packages for making pytorch models for climate and health.


Installation
-------------

Using uv (recommended):

- Clone repository
- Install with uv: ``uv sync`` (installs production dependencies)
- For development: ``uv sync --dev`` (includes test dependencies like pytest and hypothesis)
- Run the CLI: ``uv run chtorch``

Using pip:

- Clone repository
- Install with pip: ``pip install .``
- For local development: ``pip install -e ".[dev]"``
