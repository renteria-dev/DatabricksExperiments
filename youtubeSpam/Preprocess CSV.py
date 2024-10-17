# Databricks notebook source
# Load CSV file into a Spark DataFrame
df = spark.read.option("header", "true") \
               .option("inferSchema", "true") \
               .csv("file:/Workspace/Users/luis.renteria@encora.com/ML-Datasets/Youtube-Spam-Dataset.csv")

display(df)

# COMMAND ----------

df.write.saveAsTable("default.youtube_spam_dataset")
