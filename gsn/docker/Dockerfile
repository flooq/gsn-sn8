# Upgraded version from 22 to 23 due to the lack of RTX4090 support
FROM nvcr.io/nvidia/pytorch:23.03-py3
MAINTAINER Jacob Arndt <arndtjw@ornl.gov>

RUN apt-get update --fix-missing && DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install libpq-dev gdal-bin libgdal-dev -y
WORKDIR /tmp

COPY requirements.txt .

RUN set -eux && pip install --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install "opencv-python-headless<4.3"

WORKDIR /gsn

