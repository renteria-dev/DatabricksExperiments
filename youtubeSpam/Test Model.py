# Databricks notebook source
import mlflow
logged_model = 'runs:/10e8556e461d4c249bf8cd8b05808f74/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


# COMMAND ----------

df_random = spark.sql("SELECT CONTENT, CLASS FROM paco.default.youtube_spam_dataset ORDER BY RAND() LIMIT 5")
display(df_random)

# COMMAND ----------

# Convert Spark DataFrame to Pandas DataFrame
df_random_pd = df_random.toPandas()

# Make predictions
predictions = loaded_model.predict(df_random_pd['CONTENT'])

# Add predictions to the DataFrame
df_random_pd['PREDICTIONS'] = predictions

# Display the DataFrame with predictions
display(df_random_pd)
