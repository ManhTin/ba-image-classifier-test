import os
import re

from PIL import Image

PATCHES = [
  (408, 158, 438, 188),
  (445, 169, 475, 199),
  (565, 155, 595, 185),
  (595, 145, 620, 170),
  (625, 170, 660, 205),
  (665, 170, 690, 195),
  (475, 199, 520, 244),
  (542, 234, 587, 279),
  (625, 249, 675, 299),
  (698, 274, 748, 324)
]

IMAGES_PATH = 'images/full/'
PATCHES_PATH = 'images/patches/'

# create list of images in IMAGES_PATH
image_list = [f for f in os.listdir(IMAGES_PATH)
  if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

def extract_patches(img_name, img):
  # Loop through PATCHES
  for i, patch in enumerate(PATCHES):
    filename = os.path.splitext(img_name)[0]
    ext = os.path.splitext(img_name)[1]
    patch_file_name = filename + '_patch_' + str(i) + ext

    patch = img.crop(patch)
    patch.save(PATCHES_PATH + patch_file_name)

for image in image_list:
  img_name = image
  img = Image.open(IMAGES_PATH + image)
  extract_patches(img_name, img)

print('Finished extracting patches')
