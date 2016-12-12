# Rapid-Rate
UWaterloo Course Project - CS846 Software Engineering in Big Data

This project aims to evaluate the performance of shallow neural-network learned word-embeddings producing by Paragraph Vectors (Doc2Vec) for a Continuous function learning (regression) task.

## Pre-requisites
* Python 2.7+
* Gensim

## Usage Instructions


### Clone this repository

```
git clone https://github.com/v1n337/rapid-rate.git
```


### Training the document vectors and machine learning models

Run the below command, after filling in the placeholders

```
export PYTHONPATH=<PATH_TO_PROJECT> &&
/usr/bin/python <PATH_TO_PROJECT>/src/rapid-rate/rapid_rate.py
--input_file_path <PATH_TO_LABELLED_DATA_FILE>
--doc2vec_training_count <COUNT_OF_LABELLED_DATA_SHUFFLES>
--doc2vec_model_path <PATH_TO_SAVE_DOC2VEC_MODEL>
--ml_model_path <PATH_TO_SAVE_ML_MODEL>
```


### Setting up a OpenTSDB and Bosun

OpenTSDB + Bosun must be set-up on localhost, port 8070, for the default *kafka_rating_engine.py* script, but this can be altered in the script if they need to be set up elsewhere.

The guide can be found at this link
https://bosun.org/quickstart


### Setting up a Spark Streaming program to predict review ratings in real-time

Once the model is trained, the below command can be run to initiate a spark instance to predict value of unseen text snippets that are fed to a kafka producer. Zookeeper is a pre-requisite for Kafka, and the variables ZOOKEEPER_HOST and ZOOKEEPER_PORT should mirror the Zookeeper config read by Kafka.

```
./bin/spark-submit <PATH_TO_PROJECT>/src/rapid-rate/utils/kafka_rating_engine.py
<ZOOKEEPER_HOST>:<ZOOKEEPER_PORT> <KAFKA_TOPIC_NAME>
<PATH_TO_SAVED_DOC2VEC_MODEL> <PATH_TO_SAVED_ML_MODEL>
```
