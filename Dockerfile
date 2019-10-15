FROM tensorflow/tensorflow:2.0.0-gpu-py3
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y --no-install-recommends python3-tk
COPY requirements.txt /requirements.txt 
RUN pip install -r requirements.txt
COPY . .
CMD ["bash"]