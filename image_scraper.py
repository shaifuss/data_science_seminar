#!/usr/bin/env python3
import json
import os
import multiprocessing
import requests

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"


def download_review_photos(session, review_filepath, output_dir):
    with open(review_filepath, "r") as f:
        review = json.load(f)

    for i, review_photo_url in enumerate(review):
        with open(os.path.join(output_dir, f"{i}.jpg"), "wb") as f:
            with session.get(review_photo_url, stream=True) as r:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)


def download_review(review_filename):
    if not review_filename.endswith(".json"):
        return
    session = requests.session()
    session.headers["User-Agent"] = USER_AGENT
    review_id = os.path.splitext(review_filename)[0]
    output_dir = os.path.join("review_photos", review_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        download_review_photos(
            session, os.path.join("review_photos", review_filename), output_dir
        )


def main():
    with multiprocessing.Pool(6) as pool:
        pool.map(download_review, os.listdir("review_photos"))


if __name__ == "__main__":
    main()

