# PBMF - Tree distilation app
Tree distilation app for visualizing biomarkers and defining sub-populations. 

## minimal setup 
create a virtual environment
``` bash
python3 -m env pbmf_env 
source pbmf_env/bin/activate
```

You need the following packages as a minimum requirement.
``` bash
pip3 install marimo==0.9.20
pip3 install scikit-learn==1.5.2
pip3 install seaborn
pip3 install lifelines==0.30.0
pip3 install tensorflow
pip3 install tqdm
pip3 install torch
pip3 install git+https://github.com/gaarangoa/pbmf.git
pip3 install git+https://github.com/gaarangoa/samecode.git
```

## Launch marimo app
This should start the marimo app
```python3 -m marimo run app.py``` 

