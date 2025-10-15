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



Run with chapkit
--------------------
uvicorn main_chapkit:app --port 8002


Docker
------
This repo contains a Dockerfile for building a Docker image with a rest api for the model.

Build the Docker image::

    docker build -t chtorch-api .

Run the container (maps internal port 8000 to external port 8002)::

    docker run -p 8002:8000 chtorch-api

Or run in detached mode with a name::

    docker run -d --name chtorch-api -p 8002:8000 chtorch-api

Check logs::

    docker logs chtorch-api

Stop and remove container::

    docker stop chtorch-api
    docker rm chtorch-api

Docker Compose
--------------

To use the image in a docker-compose.yml file::

    version: '3.8'
    
    services:
      chtorch-api:
        image: ghcr.io/dhis2-chap/chtorch:latest  # Replace with your GitHub username
        ports:
          - "8002:8000"  # Maps external port 8002 to internal port 8000
        volumes:
          - chtorch-data:/app/target  # Persist the database
        restart: unless-stopped
    
    volumes:
      chtorch-data:

Run with docker-compose::

    docker-compose up -d

The API will be accessible at http://localhost:8002

GitHub Container Registry
-------------------------

The GitHub Actions workflow automatically builds and pushes Docker images to GitHub Container Registry (ghcr.io) on:

- Push to main/master/chapkit branches
- Version tags (v*)
- Manual workflow dispatch

Images are tagged with:

- Branch name (e.g., ``ghcr.io/username/chtorch:main``)
- Version tags (e.g., ``ghcr.io/username/chtorch:v1.0.0``)
- Latest tag for the default branch
- SHA-based tags for traceability

To pull the image::

    docker pull ghcr.io/YOUR_GITHUB_USERNAME/chtorch:latest