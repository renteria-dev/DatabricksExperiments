# Databricks notebook source
# MAGIC %md
# MAGIC Load Model without register it

# COMMAND ----------

import mlflow

model_uri = f"runs:/8cd62b954a004ae994844e03009a6e7d/model"
model = mlflow.pyfunc.load_model(model_uri=model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC Load data from table

# COMMAND ----------

df = spark.sql("SELECT CONTENT, CLASS FROM paco.default.youtube_spam_dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC Select a random comment

# COMMAND ----------

from pyspark.sql.functions import rand

random_row = df.orderBy(rand()).limit(1)
display(random_row)

# COMMAND ----------

# MAGIC %md
# MAGIC Predict if comment is spam using the trained model

# COMMAND ----------

random_row_pandas = random_row.select('CONTENT').toPandas()
prediction = model.predict(random_row_pandas)
actual_class = random_row.select('CLASS').collect()[0][0]
result = "correct" if prediction[0] == actual_class else "incorrect"
display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we are going to create a comment and pass it to the model

# COMMAND ----------

import pandas as pd

def predict_comments(new_comment):
    # Check the type of input and ensure it's always treated as a list
    if isinstance(new_comment, str):
        new_comment_list = [new_comment]
    elif isinstance(new_comment, list):
        new_comment_list = new_comment
    else:
        raise ValueError("Input must be a string or a list of strings.")
    
    # Create a DataFrame from the list of comments
    new_comment_pandas = pd.DataFrame({
        'CONTENT': new_comment_list
    })
    
    # Predict using the model (assuming model is already defined)
    new_predictions = model.predict(new_comment_pandas)
    
    # Create a result DataFrame to hold comments and their predictions
    results = []
    for comment, prediction in zip(new_comment_list, new_predictions):
        result = "IS SPAM" if prediction == '1' else "IS NOT SPAM"
        results.append((comment, result))
    
    # Create a DataFrame from the results
    result_df = pd.DataFrame(results, columns=['Comment', 'Prediction'])
    
    # Display the result DataFrame
    display(result_df)
    
    return result_df


# COMMAND ----------

# Spam comments
spam_comments = [
    "I'm making money investing on Bitcoin.",
    "This video is great! Visit my channel for free money!",
    "You can make thousands from home! DM me for details."
]

# Not spam comments
not_spam_comments = [
    "Great video, can't wait for more!",
    "I really enjoyed the tips shared in this tutorial.",
    "Thanks for sharing this!",
    "Hahaha 😂🤣  "
]

# Combine both into new_comments_array
new_comments_array = spam_comments + not_spam_comments
predict_comments(new_comments_array)

