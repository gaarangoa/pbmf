# Predictive Biomarker Modeling Framework (PBMF) 
The PBMF is an automated neural network framework based on contrastive learning. This general-purpose framework explores potential predictive biomarkers in a systematic and unbiased manner.

![alt text](./track.gif) Under the hood, the PBMF searches for a biomarker that maximizes the benefit under treatment of interest while at the same time minimizes the effect of the control treatment.

## Quick tour
The PBMF runs as follows: 

```python

from PBMF.attention.model_zoo.SimpleModel import Net
from PBMF.attention.model_zoo.Ensemble import EnsemblePBMF

# Setup ensemble
pbmf = EnsemblePBMF(
    time=time, 
    event=event,
    treatment=treatment,
    stratify=treatment,
    features = features,
    discard_n_features=1, # discard n features on each PBMF model
    architecture=Net, # Architecrture to use, we are using a simple NN.
    **params
)

# Train ensemble model
pbmf.fit(
    data_train, # Dataframe with the processed data
    num_models=10, # number of PBMF models used in the ensemble
    n_jobs=4,
    test_size=0.2, # Discard this fraction (randomly) of patients when fiting a PBMF model
    outdir='./runs/experiment_0/',
    save_freq=100,
)

```

Once the model is trained, get the predictive biomarker scores and labels is as simple as:
```python
# Load the ensemble PBMF
pbmf = EnsemblePBMF()
pbmf.load(
    architecture=Net,
    outdir='./runs/experiment_0/',
    num_models=10,
)

# Retrieve scores for predictive biomarker positive / negative
data_test['predictive_biomarker_risk'] = pbmf.predict(data_test, epoch=500)
# Generate biomarker positive and negative labels
data_test['predicted_label'] = (data_test['predictive_biomarker_risk'] > 0.5).replace([False, True], ['B-', 'B+'])

```
### PBMF demos
* Under <code>./demos/</code> you will find a complete guide on how to use the framework. 
* under <code>./demos/app</code> you can find the app for visualizing the distilation trees and * interpretability. 
* under <code>./demos/simulation</code> we have an example on how to build synthetic survival datasets.

## System Requirements
### Hardware requirements
The <code>PBMF</code> can be run in standard computers with enough RAM memory. PBMF is efficient when running on multiple cores to perform parallel trainings when setting a large number of models (<code>num_models</code>). 

The PBMF runs in <code>Python > 3</code> and has been tested on MacOS and Linux Ubuntu distributions. 

### Software requirements
This python package is supported for macOS and Linux. The PBMF has been tested on the following systems using docker and singularity containers:

#### OS requirements
* macOS: Sonoma
* Linux: Ubuntu 18.04 LTS
* Windows: WSL2 / ubuntu / x86_64


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
### Docker container
The easiest way to get started with the PBMF is to run it through a docker container. We have created an image with all necessary libraries and these containers should seamlessly work.

#### For macOS ARM processors:
```bash
    # Download the PBMF repository
    git clone https://github.com/gaarangoa/pbmf.git
    cd ./pbmf/

    # Build the docker image
    docker pull gaarangoa/ml:v2.1.0.1_ARM
    docker build -f Dockerfile.arm . --tag pbmf

    # Launch a jupyter notebook
    docker run -it --rm -p 8888:8888 pbmf jupyter notebook --NotebookApp.default_url=/lab/ --ip=0.0.0.0 --port=8888 --allow-root

```

##### For x86-64 processors:
```bash
    # Download the PBMF repository
    git clone https://github.com/gaarangoa/pbmf.git
    cd ./pbmf/

    # Build the docker image
    docker pull gaarangoa/dsai:version-2.0.3_tf2.6.0_pt1.9.0
    docker build -f Dockerfile.x86-64 . --tag pbmf

    # Launch a jupyter notebook
    docker run -it --rm -p 8888:8888 pbmf jupyter notebook --NotebookApp.default_url=/lab/ --ip=0.0.0.0 --port=8888 --allow-root
```

### Dependencies for manuscript experiments
All experiments in the manuscript were performend in our internal HCP. We used multiple nodes with 100 cores for running the PBMF in parallel. No GPU acceleration was enabled. The HCP used <code>Ubuntu 18.04</code>. For each run we deployed docker containers using <code>singularity version=3.7.1</code> the image used is available at docker hub (<code>gaarangoa/dsai:version-2.0.3_tf2.6.0_pt1.9.0</code>).


## License
The code is freely available under the MIT License
