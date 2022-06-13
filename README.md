# Single2Multi_LOMP
# Single to Multi: Data-Driven High Resolution Calibration Method for Piezoresistive Sensor Array

## Code
This repository contains the code for model introduced in  IEEE Robotics and Automation Letters [paper] (https://ieeexplore.ieee.org/document/9394718). 

# Prerequisites
It requires that you download the data from [(coming soon)]. 
or contact authors for multi-touch dataset. 
(We append a sampled single touch dataset that can be used to train our model)

```commandline
conda create -n LoMP python=3.6
pip3 install torch torchvision torchaudio
conda install 
pip install -U scikit-learn
pip install pandas
pip install matplotlib
```


Next, make sure to update the path to where you sotred the data in train_test_.py file

# Run training
With running train_test_passing_single_new.py file, you can simply type

```commandline
python train_test_passing_single_new.py
```

This loads a single touch dataset to train and test our dataset. 

# testing for multi touch dataset
Before you run the code, please make sure that you uncomment double touch loading code in train_test_passing_single_new.py file. 
Since the volume for multi touch dataset is huge, please make a contact or access NMAIL database to download the multi-touch dataset. 
