#@title Load the keras models
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model


#@title Define the custom predict function for WIT

# This function extracts 'image/encoded' field, which is a reserved key for the 
# feature that contains encoded image byte list. We read this feature into 
# BytesIO and decode it back to an image using PIL.
# The model expects an array of images that are floats in range 0.0 to 1.0 and 
# outputs a numpy array of (n_samples, n_labels)
# Use https://github.com/PAIR-code/what-if-tool/blob/3be0f74a02843e06d747a60823e50b4295019f6a/utils/inference_utils.py#L173 service_bundle object to customize the prediction model by passing metadat
def custom_predict_fn(examples,serving_bundle):
  def load_byte_img(im_bytes):
    buf = BytesIO(im_bytes)
    return np.array(Image.open(buf), dtype=np.float64) / 255.
  if "dbfs" in serving_bundle.inference_address:
    model1 = load_model(serving_bundle.inference_address)
  else:
    # Load default model
    model1 = load_model('/dbfs/user/puneet.jain@databricks.com/wit_examples/smile-model.hdf5')
  ims = [load_byte_img(ex.features.feature['image/encoded'].bytes_list.value[0]) 
         for ex in examples]
  preds = model1.predict(np.array(ims))
  return preds