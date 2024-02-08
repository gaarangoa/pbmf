# Predictive Biomarker Modeling Framework (PBMF) 
The PBMF is an automated neural network framework based on contrastive learning. This general-purpose framework explores potential predictive biomarkers in a systematic and unbiased manner.

![alt text](./track.gif) Under the hood, the PBMF searches for a biomarker that maximizes the benefit under treatment of interest while at the same time minimizes the effect of the control treatment.


## System Requirements
### Hardware requirements
The <code>PBMF</code> can be run in standard computers with enough RAM memory. PBMF is efficient when running on multiple cores to perform parallel trainings when setting a large number of models (<code>num_models</code>). 

The PBMF runs in <code>Python > 3</code> and has been tested on MacOS and Linux Ubuntu distributions. 

### Software requirements
This python package is supported for macOS and Linux. The PBMF has been tested on the following systems using docker and singularity containers:

#### OS requirements
* macOS: Sonoma
* Linux: Ubuntu 18.04 LTS


#### Python dependencies
PBMF was extensively tested using the following libraries:

```bash
tensorflow==2.6.0
scipy==1.5.4
numpy==1.19.5
scikit-learn==0.24.1
pandas==1.1.5
seaborn==0.11.1
```

The PBMF has been also tested with latest updates of the listed libraries.

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
    docker build -f Dockerfile.arm . --tag pbmf
    docker run -it --rm -p 8888:8888 pbmf jupyter notebook --NotebookApp.default_url=/lab/ --ip=0.0.0.0 --port=8888 --allow-root

```

##### x86-64 based processors
```bash
    docker build -f Dockerfile.x86-64 . --tag pbmf
    docker run -it --rm -p 8888:8888 pbmf jupyter notebook --NotebookApp.default_url=/lab/ --ip=0.0.0.0 --port=8888 --allow-root
```

### Requirements for experiments
All experiments in the manuscript were performend in our internal HCP. We used multiple nodes with 100 cores for running the PBMF in parallel. No GPU acceleration was enabled. The HCP used <code>Ubuntu 18.04</code>. For each run we deployed docker containers using <code>singularity version=3.7.1</code> the image used is available at docker hub (<code>gaarangoa/dsai:version-2.0.3_tf2.6.0_pt1.9.0</code>). Environment seed was set to 0. 