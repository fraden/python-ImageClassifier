import json
from typing import Tuple, Union, List

from PIL import Image
import numpy as np
import tensorflow as tf

from config import config


IMG_SHAPE = config['IMG_SHAPE']


def load_json(category_names: str) -> dict:
    """
    Loads a file containing category names and returns a dictionary containing the values
    Args:
        category_names (str): contains the path to the mapping file

    Returns:
        dict: containing the mappings
    """
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    return class_names


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to a tensor, resizes it and normalizes it to values between 0...1.
    Args:
        image (np.ndarray): image that should be processed

    Returns:
        np.ndarray: containing the image
    """
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (IMG_SHAPE, IMG_SHAPE))
    image /= 255
    return image.numpy()


def predict(image_path: str, model, top_k: int, category_names: str) -> Tuple[
        List[float], Tuple[Union[List[int], List[str]]]]:
    """
    Predicts the name of a flower by using an image of the specific flower.
    Args:
        image_path (str): path to the image
        model: tensorflow model that should be used for prediction
        top_k (int): number of most likely classes that should be returned
        category_names (str): Path to a JSON file mapping labels to flower names

    Returns:
        Tuple[List[float], Tuple[Union[List[int], List[str]]]]: List containing either the number of the classes or
            their names.
    """
    image = np.asarray(Image.open(image_path))
    image = process_image(image)
    predictions = model.predict(np.expand_dims(image, axis=0))[0]
    top_elements = np.argsort(predictions)[::-1][:top_k]
    if category_names is None:
        classes = top_elements
    else:
        class_names = load_json(category_names)
        classes = [class_names[str(numpy_label + 1)] for numpy_label in top_elements]
    return [predictions[i] for i in top_elements], classes
