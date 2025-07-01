FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y pandoc make && \
    apt-get clean

RUN pip install --no-cache-dir \
    sphinx \
    nbsphinx \
    myst-parser \
    ipykernel \
    ipython \
    sphinx-rtd-theme
