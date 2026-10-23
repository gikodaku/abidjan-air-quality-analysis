FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 AAQA_ROOT=/app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[api,dashboard,assistant]"

COPY config ./config
COPY dashboard ./dashboard
COPY .streamlit ./.streamlit
RUN mkdir -p data/raw data/processed models reports/figures \
    && useradd --create-home --uid 1000 aaqa && chown -R aaqa /app
USER aaqa

# On start-up: download the data and train if no model is present
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

EXPOSE 8000 8501 8502
HEALTHCHECK --interval=60s --timeout=10s --start-period=600s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "aaqa.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
