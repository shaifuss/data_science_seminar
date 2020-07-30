#!/usr/bin/env python3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
import numpy as np
import os


def load_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def main():
    model = Xception(weights="imagenet")
    dir_path = "review_photos"
    images_to_check = []
    for dirpath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if not filename.endswith(".jpg"):
                continue
            image_path = os.path.join(dirpath, filename)
            images_to_check.append(image_path)
            if len(images_to_check) >= 50:
                x = np.vstack(
                    [load_image(image_to_check) for image_to_check in images_to_check]
                )

                preds = model.predict(x)
                for image_index, pred in enumerate(decode_predictions(preds, top=5)):
                    # decode the results into a list of tuples (class, description, probability)
                    # (one such list for each sample in the batch)
                    predicted_classes = [prediction[1] for prediction in pred]
                    if "pizza" not in predicted_classes:
                        os.remove(images_to_check[image_index])
                images_to_check = []

    if len(images_to_check) > 0:
        x = np.vstack(
            [load_image(image_to_check) for image_to_check in images_to_check]
        )

        preds = model.predict(x)
        for image_index, pred in enumerate(decode_predictions(preds, top=5)):
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            predicted_classes = [prediction[1] for prediction in pred]
            if "pizza" not in predicted_classes:
                os.remove(images_to_check[image_index])


if __name__ == "__main__":
    main()
