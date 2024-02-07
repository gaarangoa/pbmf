# Predictive Biomarker Modeling Framework (PBMF) 
The PBMF is an automated neural network framework based on contrastive learning. This general-purpose framework explores potential predictive biomarkers in a systematic and unbiased manner.


![alt text](./track.gif) Under the hood, the PBMF searches for a biomarker that maximizes the benefit under treatment of interest while at the same time minimizes the effect of the control treatment.



## System Requirements

### Hardware requirements
### Software requirements
#### OS requirements
#### Python dependencies

## Installation guide
### Install from Github

## setting up the environment
### Docker container

#### Build the docker image
```bash
    git clone https://github.com/gaarangoa/pbmf.git
    cd ./pbmf/
```
##### ARM M1 processors
```bash
    docker build . --tag pbmf

    docker run -it --rm -p 8888:8888 pbmf jupyter notebook --NotebookApp.default_url=/lab/ --ip=0.0.0.0 --port=8888 --allow-root

```

##### x86-64 based processors
```bash
    docker build . --tag pbmf

    docker run -it --rm -p 8888:8888 pbmf jupyter notebook --NotebookApp.default_url=/lab/ --ip=0.0.0.0 --port=8888 --allow-root
```

