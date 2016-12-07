from __future__ import print_function

import sys

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


def infer_sentiment(rdd):
    for (key, message) in rdd.collect():

        if message:
            tokenized_message = tokenize_text(str(message))
            message_vector = \
                d2v_model.infer_vector(doc_words=tokenized_message)
            prediction = ml_model.predict([message_vector])
            print(round(prediction[0], 2))
        else:
            pass  # ignore if no message present


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
