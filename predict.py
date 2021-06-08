import argparse

from keras.models import load_model
from utilities import utilities

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('image', type=str, help='image path')
parser.add_argument('model', type=str, help='path to model')
parser.add_argument('--top_k', type=int, help='number of top elements', default=5)
parser.add_argument('--category_names', type=str, help='path to json file, that maps labels to flower names')
args = parser.parse_args()

top_k = args.top_k
category_names = args.category_names


model = load_model(args.model)
predictions, classes = utilities.predict(args.image, model, top_k, category_names)

print(classes)
