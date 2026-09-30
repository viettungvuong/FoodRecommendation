FROM python:3.12-slim

WORKDIR /opt/remymy-api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/remymy-api/src \
    HOST=0.0.0.0 \
    PORT=8002 \
    DATABASE_PATH=/data/remymy-food.db \
    USER_DATABASE_PATH=/state/remymy-users.db

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY migrations ./migrations
COPY README.md .

EXPOSE 8002

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '8002') + '/ready', timeout=2)" || exit 1

CMD ["sh", "-c", "exec uvicorn remymy_api.main:app --host ${HOST} --port ${PORT}"]
