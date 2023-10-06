# TFN-ROMs
The broad code implementation and dataset files corresponding to the study in this repository are available to the reader who needs them

# Python libraries & version control
torch                     1.12.1+cu116

torchvision               0.13.1+cu116

torchaudio                0.12.1+cu116

numpy                     1.21.6

pandas                    1.2.4

python                    3.9.12

scikit-learn              1.0.2

# Introduction to the documents
│  data.xlsx
│  list.txt
│  
├─.idea
│      modules.xml
│      TFN-ROMs.iml
│      workspace.xml
│      
├─code
│  │  get_data.py
│  │  loss.py
│  │  main.py
│  │  model.py
│  │  train.py
│  │  utils.py
│  │  
│  ├─checkpoint
│  │      0.715_0.967_last_model.ckpt
│  │      
│  └─__pycache__
│          get_data.cpython-39.pyc
│          loss.cpython-39.pyc
│          model.cpython-39.pyc
│          train.cpython-39.pyc
│          utils.cpython-39.pyc
│          
├─params
│      params.py
│      
└─__pycache__

data.xlsx： The dataset used in this study

get_data.py: Importing data and processing

loss.py: Customize the objective function

main.py: Master Functions

model.py: Model structure

train.py: Training Functions

utils.py: Common Functions

0.715_0.967_last_model.ckpt: Pre-trained model weights

params.py: Parameter summary
