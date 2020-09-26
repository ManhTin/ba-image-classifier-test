import time
import os
import re
import numpy as np

from PIL import Image
import tensorflow as tf
# from tflite_runtime.interpreter import tflite

LABELS = 'model/labels.txt'
MODEL = 'model/park_mobilenet_v2_100_224_lite.tflite'
PATCHES_PATH = 'images/patches'
CLASSES = ['empty', 'occupied']

interpreter = tf.lite.Interpreter(MODEL)
interpreter.allocate_tensors()

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

# predict class of image
def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

# iterate files
def iterate_files(path, true_label):
  image_list = [f for f in os.listdir(path)
    if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

  for image in image_list:
    img = Image.open(path + '/' + image).resize((224,224))
    #do sth.
    start_time = time.time()
    predicted_labels = classify_image(interpreter, img)
    elapsed_ms = (time.time() - start_time) * 1000
    predicted_label_id, prob = predicted_labels[0]

    durations.append(elapsed_ms)
    probabilities.append(prob)

    if predicted_label_id == true_label:
      errors+=1

def main():
  labels = load_labels(LABELS)

  errors = 0
  durations = []
  probabilities = []

  # Iterate folders
  for idx, class_name in enumerate(CLASSES):
    print("Iterating {} folder".format(class_name))
    path = os.path.join(PATCHES_PATH, class_name)
    iterate_files(path, idx)

  accuracy = (1 - (errors / 500)) * 100
  average_latency = np.mean(durations)
  average_probability = np.mean(probabilities)

  print("Accuracy: {}%, average latency: {} ms, average probability: {} %".format(accuracy, average_latency, average_probability))

main()
