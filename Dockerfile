# syntax=docker/dockerfile:1.0.0-experimental

# Build command requires docker build kit and ssh key path
# DOCKER_BUILDKIT=1 docker build --ssh github_ssh_key=<ssh_key_path> -t nsrdb -f Dockerfile .

FROM python:3.8-slim-buster

WORKDIR /nsrdb
RUN mkdir -p -m 755 /nsrdb

# Install aws-lambda-cpp build dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    git \
    openssh-server \
    openssh-client

# Copy package
COPY . /nsrdb

# Download public key for github.com
RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN --mount=type=ssh,id=github_ssh_key pip install --no-cache-dir git+ssh://git@github.com/NatLabRockies/rest2.git
RUN pip install --no-cache-dir -e .
RUN --mount=type=ssh,id=github_ssh_key pip install --no-cache-dir git+ssh://git@github.com/NatLabRockies/mlclouds.git

ENTRYPOINT ["nsrdb"]
