# syntax=docker/dockerfile:1
# CPU image: React build + C++/OpenMP engine + FastAPI, one container.
#   docker build -f docker/Dockerfile -t flipster .
#   docker run -p 8000:8000 flipster

FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml CMakeLists.txt README.md LICENSE ./
COPY core core
COPY bindings bindings
COPY python python
RUN FLIPSTER_ENABLE_CUDA=OFF pip wheel --no-deps -w /wheels .

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir "$(ls /wheels/flipster-*.whl)[server]" && rm -rf /wheels
COPY server server
COPY samples samples
COPY --from=web /web/dist web/dist
ENV FLIPSTER_DATA_DIR=/data \
    FLIPSTER_SAMPLES_DIR=/app/samples/scans \
    FLIPSTER_WEB_DIST=/app/web/dist
VOLUME /data
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--app-dir", "server", "--host", "0.0.0.0", "--port", "8000"]
