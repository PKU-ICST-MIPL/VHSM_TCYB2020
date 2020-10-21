# Description

This is the source code of our IEEE TCYB 2020 paper "Visual-textual Hybrid Sequence Matching for Joint Reasoning". Please cite the following paper if you use the resources.

Xin Huang, Yuxin Peng and Zhang Wen, "Visual-textual Hybrid Sequence Matching for Joint Reasoning", IEEE Transactions on Cybernetics (TCYB), 2020.

## Environment
	python 2.7
	tensorflow 1.5

## Data Preparation

You can download the [SNLI] (https://drive.google.com/file/d/1CxjKsaM6YgZPRKmJhNn7WcIC3gISehcS/view?usp=sharing) dataset and [Flickr30K] images (http://shannon.cs.illinois.edu/DenotationGraph/data/index.html) used in our paper. All the data files should be unzipped and saved in directory data/ (You need to create the directory data under directory VHSM).

## Training 
$ sh run.sh
## Testing
$ sh test.sh
You can edit the configuration variables in the training and testing scripts to your own setting. After running these scripts, you can train the model and obtain the testing results. 
