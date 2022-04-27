# syntax=docker/dockerfile:1
FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND noninteractive
#RUN apt update -y && apt upgrade -y
#RUN apt install software-properties-common -y
#RUN add-apt-repository ppa:deadsnakes/ppa -y
#RUN apt install python3.8 python3-pip -y
RUN apt-get update
RUN apt-get install -y python3.7
RUN apt-get install -y python3-pip
#RUN apt install numpy -y
RUN pip3 install tensorflow
RUN pip3 install scikit-image
RUN pip3 install keras
RUN pip3 install pillow
RUN pip3 install imagehash
COPY test.py .
COPY model.py .
COPY data.py .
COPY test_png ./test_png/
COPY data ./data/
COPY new_model.hdf5 .
CMD ["python3", "test.py"]

