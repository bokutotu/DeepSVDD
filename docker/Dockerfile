FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

ENV HOME=/home

COPY requirements.txt /requirements.txt
RUN apt-get update -y && \
    apt-get upgrade --fix-missing --fix-broken -y && \
    apt-get install libgtk2.0-dev -y && \
    conda install -c conda-forge mlflow==1.8.0 && \
    conda install -c conda-forge hydra && \
    pip install torchsummary

