FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -e .

ENV REPO2ENV_DEFAULT_REPO=/app/examples/repo_a

EXPOSE 8000

CMD ["repo2env-openenv-server", "--port", "8000"]
