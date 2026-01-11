FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /usr/src/app/requirements.txt

RUN pip install --no-cache-dir -r /usr/src/app/requirements.txt

COPY src/app.py /usr/src/app/app.py

RUN mkdir -p /usr/src/app/logs

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app", "--workers", "1", "--threads", "4", "--timeout", "120"]
