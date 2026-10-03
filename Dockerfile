FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SYNTHPANEL_DATA_DIR=/app/data

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY configs ./configs
COPY scripts ./scripts
ENV SYNTHPANEL_CONFIG_DIR=/app/configs

EXPOSE 8090
ENTRYPOINT ["sh", "scripts/entrypoint.sh"]
