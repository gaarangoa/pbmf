# PBMF 
This repository contains the instructions to test the PBMF

![alt text](./track.gif)

## Run notebook and explore results

```bash

git clone https://github.com/gaarangoa/pbmf.git
cd ./pbmf/

docker build . --tag pbmf

docker run -it --rm -p 8888:8888 pbmf jupyter notebook --NotebookApp.default_url=/lab/ --ip=0.0.0.0 --port=8888 --allow-root

```

