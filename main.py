import time
import os
import re
import numpy as np

from PIL import Image
# import tensorflow as tf # uncomment for local development
import tflite_runtime.interpreter as tflite # comment for local development

LABELS = 'model/labels.txt'
MODEL = 'model/park_mobilenet_v2_100_224_lite.tflite'
PATCHES_PATH = 'images/patches'
CLASSES = ['empty', 'occupied']

# interpreter = tf.lite.Interpreter(MODEL) # uncomment for local development
interpreter = tflite.Interpreter(MODEL) # comment for local development
interpreter.allocate_tensors()

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}

labels = load_labels(LABELS)

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  image_tensor = interpreter.tensor(tensor_index)()[0]
  image_tensor[:, :] = image

# predict class of image
def classify_image(interpreter, image, top_k=1):
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output_index = output_details['index']
  output = np.squeeze(interpreter.get_tensor(output_index))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)
    print(output_details['dtype'])

  ordered = np.argpartition(-output, top_k)
  ordered_output = [(i, output[i]) for i in ordered[:top_k]]
  return ordered_output[0]

# iterate files
def iterate_files(path, true_label, errors, durations):
  image_list = [f for f in os.listdir(path)
    if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

  for image in image_list:
    img = Image.open(path + '/' + image).resize((224,224))
    start_time = time.time()
    predicted_labels = classify_image(interpreter, img)
    elapsed_ms = (time.time() - start_time) * 1000
    predicted_label_id = predicted_labels[0]

    durations.append(elapsed_ms)
    if predicted_label_id != true_label:
      errors +=1

def main():
  errors = 0
  durations = []
  # Iterate folders
  for idx, class_name in enumerate(CLASSES):
    path = os.path.join(PATCHES_PATH, class_name)
    iterate_files(path, idx, errors, durations)

  accuracy = (1 - (errors / 500)) * 100
  average_latency = round(np.mean(durations), 2)
  standard_deviation = round(np.std(durations), 2)
  print("Performed {} predictions. Accuracy: {}%, average latency: {} ms, standard deviation: {} ms".format(len(durations), accuracy, average_latency, standard_deviation))

print('Starting...')

for x in range(1,6):
  main()

print('Finished.')
