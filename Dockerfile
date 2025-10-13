# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for building Python packages and git
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
RUN pip install --no-cache-dir uv

# Copy pyproject.toml and uv.lock first for better layer caching
COPY pyproject.toml ./
# Note: uv.lock might need to be regenerated after changing to GitHub sources
# If the build fails, you may need to run 'uv lock' locally first

# Copy the entire codebase
COPY chtorch ./chtorch
COPY main_chapkit.py ./
COPY README.rst ./

# Install dependencies using uv sync
RUN uv sync --no-dev

# Create directory for the database
RUN mkdir -p target

# Expose the port
EXPOSE 8000

# Set environment variable to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Run the application
CMD ["uvicorn", "main_chapkit:app", "--host", "0.0.0.0", "--port", "8000"]
