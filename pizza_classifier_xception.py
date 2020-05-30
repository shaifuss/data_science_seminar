#!/usr/bin/env python3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
import numpy as np
import os


def main():
    model = Xception(weights="imagenet")
    dir_path = "review_photos"
    for dirpath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if not filename.endswith(".jpg"):
                continue
            img_path = os.path.join(dirpath, filename)
            img = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            predicted_classes = [
                prediction[1] for prediction in decode_predictions(preds, top=5)[0]
            ]
            if "pizza" not in predicted_classes:
                print(img_path)


if __name__ == "__main__":
    main()
