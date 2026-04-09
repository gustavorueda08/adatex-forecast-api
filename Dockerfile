FROM python:3.11-slim

# Build tools required by cmdstanpy to compile CmdStan (needed by Prophet)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install Python packages and pre-compile CmdStan during build time.
# Doing this at build (not runtime) means the compiled binary is cached in the
# image layer and the container starts instantly without needing to recompile.
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import cmdstanpy; cmdstanpy.install_cmdstan()" \
    && rm -rf /tmp/* /root/.cache/pip

COPY . .

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
