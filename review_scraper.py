#!/usr/bin/env python3
from collections import Counter
import itertools
import json
import multiprocessing
import os
import time
import pandas as pd
import random
from hashlib import sha1
import hmac
from base64 import b64encode
import requests
from getpass import getpass

DEFAULT_PARAMS = {
    "lang": "en",
    "cc": "US",
    "efs": "FBf+3IYS6CAr5S/dfac0ZMDUNC57RSNWrpbNKxAfHR+20LFxfHE7PQwEZX4HNHaKA9hOVVx98+bds5tikQfm0TXy8TPEJXZgw2+5fdOHsjA=",
    "limit": "1",
    "ywsid": "Y3yWooClkisSbx32yJG5Ww",
    "device_type": "samsung+j7y17lte/NRD90M",
    "offset": "0",
    "app_version": "20.27.0-21202717",
    "is_respond_to_review_eligible": "false",
}
SECRET_KEY = getpass("Enter Yelp's secret key:").encode("utf-8")


def get_review_photos(review):
    mobile_session = requests.session()
    mobile_session.headers[
        "User-Agent"
    ] = "Version/1 Yelp/v20.27.0-21202717 Carrier/none Model/j7y17lte OSBuild/NRD90M Android/7.0"
    mobile_session.headers["x-foregrounded"] = "true"
    params = DEFAULT_PARAMS.copy()
    params["business_id"] = review["business_id"]
    params["selected_review_id"] = review["review_id"]
    params["time"] = int(time.time())
    params["nonce"] = b64encode(
        ("".join(chr(random.randrange(256)) for _ in range(4))).encode("utf-8")
    ).decode("utf-8")
    hmac_data = "/reviews" + "".join(
        "%s=%s" % (param_key, params[param_key]) for param_key in sorted(params.keys())
    )
    params["signature"] = "_" + b64encode(
        hmac.new(SECRET_KEY, hmac_data.encode("utf-8"), sha1).digest()
    ).decode("utf-8")
    review_resp = mobile_session.get("https://auto-api.yelp.com/reviews", params=params)
    review_resp.raise_for_status()
    review_resp_json = review_resp.json()
    if (
        len(review_resp_json["reviews"]) == 0
        or len(review_resp_json["reviews"][0]["photos"]) == 0
    ):
        return []
    else:
        return [
            "%so%s" % (photo_data["url_prefix"], photo_data["url_suffix"])
            for photo_data in review_resp_json["reviews"][0]["photos"]
        ]


def main():
    pizza_businesses = []
    with open("yelp_academic_dataset_business.json", "r") as f:
        for line in f:
            line_json = json.loads(line)
            if "Pizza" in (line_json.get("categories") or ""):
                pizza_businesses.append(line_json)
    pizza_business_ids = set(
        [pizza_business["business_id"] for pizza_business in pizza_businesses]
    )
    pizza_reviews = []
    with open("yelp_academic_dataset_review.json", "r") as f:
        for line in f:
            line_json = json.loads(line)
            if line_json["business_id"] in pizza_business_ids:
                pizza_reviews.append(line_json)

    reviews_counts = Counter()
    tmp_reviews = []
    with multiprocessing.Pool(6) as pool:
        for line_json in pizza_reviews:
            if (line_json["stars"] in [1.0, 5.0]) and (
                reviews_counts[line_json["stars"]] < 70000
            ):
                tmp_reviews.append(line_json)
            if len(tmp_reviews) >= 6:
                batch_result = pool.map(get_review_photos, tmp_reviews)
                for i, review_photos in enumerate(batch_result):
                    if len(review_photos) == 0:
                        continue
                    reviews_counts[tmp_reviews[i]["stars"]] += 1
                    with open(
                        os.path.join(
                            "review_photos", f"{tmp_reviews[i]['review_id']}.json"
                        ),
                        "w",
                    ) as review_file:
                        json.dump(review_photos, review_file)
                tmp_reviews = []
                time.sleep(0.05)


if __name__ == "__main__":
    main()
