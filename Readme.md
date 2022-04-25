
# python-ImageClassifier
## Introduction
This repository includes my jupyter notebook (in assets) to develop the Image Classifier
using a Neural Network.

Furthermore it includes the code for the "Image Classifier - Part 2 - Command Line App"
of the Udacity Course "Introduction to Machine Learning with TensorFlow".

### Usage
To run the code you have to execute:
```
python predict.py /path/to/image saved_model
```
As the saved model you can use `my_model.h5`.

Furthermore by using
* `--top_k` you can specify the K most likely classes
* `--category_names`  JSON file mapping labels to flower names
