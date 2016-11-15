# IRateYouRate
UWaterloo Course Project - CS846 Software Engineering in Big Data

This project aims to evaluate the performance of shallow neural-network learned word-embeddings producing by Paragraph Vectors (Doc2Vec) for a Continuous function learning (regression) task.

## Pre-requistes
* Python 2.7+
* Gensim

## Usage Instructions

Clone this repository


```
git clone https://github.com/v1n337/IRateYouRate.git
```


Run the below command, after filling in the placeholders


```
export PYTHONPATH=<PATH_TO_PROJECT> && /usr/bin/python <PATH_TO_PROJECT>/src/irateyourate/irateyourate.py --input_file_path <PATH_TO_LABELLED_DATA_FILE> --doc2vec_training_count <COUNT_OF_LABELLED_DATA_SHUFFLES> --doc2vec_model_path <PATH_TO_SAVE_DOC2VEC_MODEL> --ml_model_path <PATH_TO_SAVE_ML_MODEL>
```

