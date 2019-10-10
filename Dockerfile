FROM tensorflow/tensorflow:1.15.0rc2-gpu-py3
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y --no-install-recommends python3-tk
COPY requirements.txt /requirements.txt 
RUN pip install -r requirements.txt
COPY . .
CMD ["bash"]