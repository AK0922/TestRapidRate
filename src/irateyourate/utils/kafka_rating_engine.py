from __future__ import print_function

import sys, requests, json, time, random

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from gensim.models.doc2vec import Doc2Vec
from sklearn.externals import joblib
from nltk import sent_tokenize, word_tokenize

d2v_model = None
ml_model = None


def tokenize_text(text):
    all_words = list()

    sentences = sent_tokenize(text)
    for sentence in sentences:
        all_words.extend(word_tokenize(sentence))

    return all_words


def post_to_opentsdb(payload):

    url = "http://127.0.0.1:8070/api/put"
    headers = {
        'content-type': "application/json",
    }
    response = requests.request("POST", url, data=json.dumps(payload), headers=headers)

    print("TSDB post Status Code: " + str(response.status_code))
    print("Dumped batch to TSDB")


def generate_tsdb_metric(metric, value, timestamp, tags):
    if not tags:
        tags = dict()
        tags['source'] = "unknown1"

    metric_dict = None
    if metric and value and timestamp:
        metric_dict = dict()
        metric_dict['metric'] = metric
        metric_dict['timestamp'] = timestamp
        metric_dict['value'] = value
        metric_dict['tags'] = tags

    return metric_dict


def infer_sentiment(rdd):

    metrics = list()

    for (key, message) in rdd.collect():

        if message:

            tokenized_message = tokenize_text(str(message))
            message_vector = \
                d2v_model.infer_vector(doc_words=tokenized_message)
            prediction = ml_model.predict([message_vector])

            metric_dict = generate_tsdb_metric("sentiment", round(prediction[0], 2), int(time.time() * 1000), None)
            print(round(prediction[0], 2))

            if metric_dict:
                metrics.append(metric_dict)

        else:
            pass  # ignore if no message present

    if len(metrics) > 0:
        post_to_opentsdb(payload=metrics)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: kafka_rating_engine.py <zk> <topic> <d2v_model> <ml_model", file=sys.stderr)
        exit(-1)

    zkQuorum, topic, doc2vec_model_path, ml_model_path = sys.argv[1:]

    print("Loading D2V model")
    d2v_model = Doc2Vec.load(doc2vec_model_path)

    print("Loading ML model")
    ml_model = joblib.load(ml_model_path)

    sc = SparkContext(appName="PythonStreamingKafkaWordCount")
    ssc = StreamingContext(sc, 1)

    kvs = \
        KafkaUtils.createStream(
            ssc, zkQuorum, "spark-streaming-consumer", {topic: 1}
        )

    kvs.foreachRDD(infer_sentiment)

    ssc.start()
    ssc.awaitTermination()
