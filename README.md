# PBMF demo
This repository contains the instructions to test the PBMF

## Run notebook and explore results

# Install

## Install docker

## Download this repository

##

```bash

cd ./pbmf/

docker build . --tag pbmf

docker run -it --rm -p 8888:8888 pbmf jupyter notebook --NotebookApp.default_url=/lab/ --ip=0.0.0.0 --port=8888

```