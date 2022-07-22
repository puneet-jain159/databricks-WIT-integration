# Databricks notebook source
# MAGIC %md 
# MAGIC # Demo on how to use the What-if Tool present in tensorboard with using tensorflow serving

# COMMAND ----------

# MAGIC %md
# MAGIC To use the What-If Tool inside of TensorBoard, you need to serve your model through TensorFlow Serving’s prediction service API, in which models accept TensorFlow Example protocol buffers as input data points, or you can provide your own **custom python function for generating model predictions**.
# MAGIC 
# MAGIC We will use the following link to create a custom model prediction function https://pair-code.github.io/what-if-tool/learn/tutorials/tensorboard/

# COMMAND ----------

import os
try:
  username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
except:
  username = str(uuid.uuid1()).replace("-", "")
experiment_log_dir = "/dbfs/user/{}/tensorboard_log_dir/".format(username)
loc =  "/dbfs/user/{}/wit_example".format(username)
repo_loc = os.getcwd()
print("test directory location : ",loc)
print("repo_loc", repo_loc)
os.environ['loc'] = loc
if not os.path.isdir(loc):
  print("creating director",loc)
  os.mkdir(loc)


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Import the necessary packages

# COMMAND ----------

import numpy as np
import tensorflow as tf
import os
from PIL import Image
from io import BytesIO

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download dummy dataset

# COMMAND ----------

!curl -L https://storage.googleapis.com/what-if-tool-resources/smile-demo/smile-colab-model.hdf5 -o $loc/smile-model.hdf5
!curl -L https://storage.googleapis.com/what-if-tool-resources/smile-demo/test_subset.zip -o $loc/test_subset.zip

!unzip -o $loc/test_subset.zip -d $loc/test

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create helper functions 

# COMMAND ----------


# Converts a dataframe into a list of tf.Example protos.
# If images_path is specified, it assumes that the dataframe has a special 
# column "image_id" and the path "images_path/image_id" points to an image file.
# Given this structure, this function loads and processes the images as png byte_lists
# into tf.Examples so that they can be shown in WIT. Note that 'image/encoded'
# is a reserved field in WIT for encoded image features.
def df_to_examples(df, columns=None, images_path=''):
  examples = []
  if columns == None:
    columns = df.columns.values.tolist()
  for index, row in df.iterrows():
    example = tf.train.Example()
    for col in columns:
      if df[col].dtype is np.dtype(np.int64):
        example.features.feature[col].int64_list.value.append(int(row[col]))
      elif df[col].dtype is np.dtype(np.float64):
        example.features.feature[col].float_list.value.append(row[col])
      elif row[col] == row[col]:
        example.features.feature[col].bytes_list.value.append(row[col].encode('utf-8'))
    if images_path:
      fname = row['image_id']
      with open(os.path.join(images_path, fname), 'rb') as f:
        im = Image.open(f)
        buf = BytesIO()
        im.save(buf, format= 'PNG')
        im_bytes = buf.getvalue()
        example.features.feature['image/encoded'].bytes_list.value.append(im_bytes)
    examples.append(example)
  return examples

def make_label_column_numeric(df, label_column, test):
  df[label_column] = np.where(test(df[label_column]), 1, 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert the test data into tf.records format needed for WIT Tool

# COMMAND ----------

import pandas as pd

data = pd.read_csv(f'{loc}/test/celeba/data_test_subset.csv')
examples = df_to_examples(data, images_path=f'{loc}/test/celeba/img_test_subset_resized/')

record_file = f'{loc}/test/images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
  for example in examples:
    writer.write(example.SerializeToString())

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Load the tensorboard 

# COMMAND ----------

# MAGIC %load_ext tensorboard

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Start the tensorboard 
# MAGIC 
# MAGIC Make sure to specify the following arguements
# MAGIC 
# MAGIC 1. **--logdir**  dbfs location to log the tensorboard logs
# MAGIC 2. **--whatif-use-unsafe-custom-prediction** location of the python file which contains the custom_predict_fn this is part of the repo
# MAGIC 3. **--whatif-data-dir** The directory location where the test/inference data is stored

# COMMAND ----------

# MAGIC %md
# MAGIC !(image/connection.png)

# COMMAND ----------

# MAGIC %tensorboard --logdir $experiment_log_dir --whatif-use-unsafe-custom-prediction $repo_loc/custom_func.py  --whatif-data-dir $loc/test/

# COMMAND ----------



# COMMAND ----------


