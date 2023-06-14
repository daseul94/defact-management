import re
import uuid
import json
import time
import cv2
import requests

import numpy as np
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt


def plt_imshow(title='image', img=None, figsize=(8, 5)):
    plt.figure(figsize=figsize)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []

            for i in range(len(img)):
                titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()


def put_text(image, text, x, y, color=(0, 255, 0), font_size=22):
    if type(image) == np.ndarray:
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(color_coverted)

    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)

    draw.text((x, y), text, font=font, fill=color)

    numpy_image = np.array(image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return opencv_image


def get_ocr_result(file, secret_key: str, api_url: str):
    files = [('file', file)]
    params = {
        "images": [{"format": "jpg", "name":"demo"}],
        "requestId": str(uuid.uuid4()),
        "version": "V2",
        "timestamp": int(round(time.time() * 1000))
    }
    payload = {
        'message': json.dumps(params).encode('UTF-8')
    }
    headers = {
        'X-OCR-SECRET': secret_key,
    }

    return requests.request(
        "POST",
        api_url,
        headers=headers,
        data=payload,
        files=files
    ).json()


def get_text(result: dict):
    return "\n".join([
        " ".join([v["inferText"] for v in cand["fields"]])
        for cand in result['images']
    ])

def processing(text:str):
    text = re.sub("[ \n]+", "", text)
    return text.strip()