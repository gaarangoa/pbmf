def show_config(args):
    print('''
project:
 date: Jan 23 2024
 name: POSEIDON_D419MC00004
 vault: https://astrazeneca.solvebio.com/vaults?vault=4049
 creators:
  Jon Doe: jon.doe@jdoe.com
 data_sources_path: ~/proj/PODS/GeMinAI/tasks/BioMarkers/Immunotherapy/data/
 data_sources:
  clinical: clinical.csv
  FMI: fmi.csv
  bloodRNA: blood.csv
  radiomics: poseidon.csv
  signatures: sig.csv

logger:
 run_path: /path/to/output/dir/

data:
 version: Jan 23 2024
 file: /path/to/data/file.csv
 separator: ,
 
outcomes:
 time: OS
 event: OS_EVENT
 arm_column: ARM
 treatment: SOC+D+T
 control: SOC+D
 

features:
 continuous: [AGE]
 categorical: [SMOKSTAT_Never] 
 mutations: []
 expression: []
 
pbmf:
 seed: 0
 repetitions: 1
 test_size: 0.3
 ignore_patients_frac: 0.1
 layers: [64]
 epochs: 100
 minp: 0.5
 w1: 1.0
 w2: 0.0
 embeddings_dim: 32
 learning_rate: 0.01
 shuffle: True
 shuffle_features: False
 l1: 0.0
 discard_n_features: 1
 num_models: 100
 n_jobs: 1
 ignore_samples_frac: 0.2
 save_freq: 100
 '''
    )