import sys
import os
import random
from io import StringIO
import numpy as np

from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
import boto3

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.mllib.feature import Word2Vec
from pyspark.ml.linalg import Vectors
from collections import defaultdict
from pyspark.sql import functions as F


class UdfFunction:
    @staticmethod
    def sortF(movie_list, timestamp_list):
        """
        sort by time and return the corresponding movie sequence
        eg:
            input: movie_list:[1,2,3]
                   timestamp_list:[1112486027,1212546032,1012486033]
            return [3,1,2]
        """
        pairs = []
        for m, t in zip(movie_list, timestamp_list):
            pairs.append((m, t))
        # sort by time
        pairs = sorted(pairs, key=lambda x: x[1])
        return [x[0] for x in pairs]


def processItemSequence(spark, rawSampleDataPath):
    # rating data
    ratingSamples = spark.read.format("csv").option("header", "true").load(rawSampleDataPath)
    ratingSamples.show(5)
    ratingSamples.printSchema()
    sortUdf = udf(UdfFunction.sortF, ArrayType(StringType()))
    userSeq = ratingSamples \
        .where(F.col("rating") >= 3.5) \
        .groupBy("userId") \
        .agg(sortUdf(F.collect_list("movieId"), F.collect_list("timestamp")).alias('movieIds')) \
        .withColumn("movieIdStr", array_join(F.col("movieIds"), " "))
    userSeq.select("userId", "movieIdStr").show(10, truncate = False)
    return userSeq.select('movieIdStr').rdd.map(lambda x: x[0].split(' '))


def embeddingLSH(spark, movieEmbMap):
    movieEmbSeq = []
    for key, embedding_list in movieEmbMap.items():
        embedding_list = [np.float64(embedding) for embedding in embedding_list]
        movieEmbSeq.append((key, Vectors.dense(embedding_list)))
    movieEmbDF = spark.createDataFrame(movieEmbSeq).toDF("movieId", "emb")
    bucketProjectionLSH = BucketedRandomProjectionLSH(inputCol="emb", outputCol="bucketId", bucketLength=0.1,
                                                      numHashTables=3)
    bucketModel = bucketProjectionLSH.fit(movieEmbDF)
    embBucketResult = bucketModel.transform(movieEmbDF)
    print("movieId, emb, bucketId schema:")
    embBucketResult.printSchema()
    print("movieId, emb, bucketId data result:")
    embBucketResult.show(10, truncate=False)
    print("Approximately searching for 5 nearest neighbors of the sample embedding:")
    sampleEmb = Vectors.dense(0.795, 0.583, 1.120, 0.850, 0.174, -0.839, -0.0633, 0.249, 0.673, -0.237)
    bucketModel.approxNearestNeighbors(movieEmbDF, sampleEmb, 5).show(truncate=False)


def trainItem2vecAndSave(spark, samples, embLength, embOutputPath):
    bucket = embOutputPath.split('//')[1].split('/',1)[0]
    key = embOutputPath.split('//')[1].split('/',1)[1]
    
    word2vec = Word2Vec().setVectorSize(embLength).setWindowSize(5).setNumIterations(10)
    model = word2vec.fit(samples)
    synonyms = model.findSynonyms("158", 20)
    #print(synonyms)
    for synonym, cosineSimilarity in synonyms:
        print(synonym, cosineSimilarity)
    
    buffer = StringIO()
    for movie_id in model.getVectors():
        vectors = " ".join([str(emb) for emb in model.getVectors()[movie_id]])
        buffer.write(movie_id + ":" + vectors + "\n")
    buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.read())
    
    
    embeddingLSH(spark, model.getVectors())
    return model


args = getResolvedOptions(sys.argv, ['JOB_NAME','SOURCE_PATH', 'OUTPUT_PATH'])

source_path = args['SOURCE_PATH']
output_path = args['OUTPUT_PATH']
rawSampleDataPath = source_path + "ratings.csv"
embOutputPath = output_path +'embedding/item2vecEmb.csv'
embLength = 10

glueContext = GlueContext(SparkContext.getOrCreate())
spark = glueContext.spark_session
s3 = boto3.client('s3')

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

samples = processItemSequence(spark, rawSampleDataPath)
model = trainItem2vecAndSave(spark, samples, embLength,
                          embOutputPath=embOutputPath)

job.commit()