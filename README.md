# BA Image Classifier Test

Contains script to run the model created in [Training](https://github.com/ManhTin/ba-image-classifier-training).

## Dependencies

* Python 3.7.9
* Pip
* Pipenv `pip install pipenv`

## Installation

Install packages for local development
`pipenv install --dev`

Install packages for deployment
* Install packages `pipenv install`
* Launch virtual environment where the packages are installed `pipenv shell`
* To install TensorFlow Lite Interpreter follow [TFLite Python quickstart guide](https://www.tensorflow.org/lite/guide/python) for your device

## Setup

* Unzip `images.zip` into project root folder

## Run script

* Launch the virtual environment created in the Installation step `pipenv shell`
* Run script via `python main.py` or `python3 main.py` depending on your Python installation
* To exit virtual environment press `CTRL + d` or type `exit`

## Test Data
The patches inside `images.zip` were created using `patches_generator.py` and sorted manually into the folders `empty` and `occupied`. You can do it yourself by unzipping `images.zip` and running `python patches_generator.py` inside the virtual env. Delete the contents of the folders `empty` and `occupied`. Sorting has to be done manually.
