import os
import re
import argparse
from glob import glob

import cv2
import easyocr
import torch
from tqdm import tqdm
from Levenshtein import ratio as levenshtein_score


parser = argparse.ArgumentParser("Test OCR model")
parser.add_argument(
    "--ocr_model_name_or_path",
    type=str,
    default=None,
    help="OCR 모델의 경로 또는 이름을 입력합니다."
)

parser.add_argument(
    "--data_name_or_path",
    type=str,
    default="deep-text-recognition-benchmark/sample/",
    help="학습 데이터에 대한 정보가 기술되어 있는 디렉토리 경로 입니다."
)

args = parser.parse_args()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    if args.ocr_model_name_or_path is None:
        ocr_model = easyocr.Reader(["ko", "en"], model_storage_directory=args.ocr_model_name_or_path)

    images = glob(os.path.join(args.data_name_or_path, "**/*.jpg"), recursive=True)
    with open(os.path.join(args.data_name_or_path, "gt.txt"), "r") as f:
        labels = f.readlines()
        labels = [l.split("\t") for l in labels]
    labels = dict(labels)
    labels = {k.split("\\")[-1]: re.sub(" ", "", v).strip() for k, v in labels.items()}
    print(labels)
    outputs = {}
    for path in tqdm(images, desc="Predicting..."):
        name = os.path.basename(path)
        print(name)
        image = cv2.imread(path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        output = ocr_model.readtext(image)
        output = " ".join([o[1] for o in output])
        output = re.sub(" ", "", output)
        outputs[name] = output

    predictions, references = [], []
    for name in outputs:
        label = labels.get(name)
        if label is None:
            continue
        predictions.append(outputs[name])
        references.append(label)
    print(predictions)
    print(references)
    print("Accuracy(Exact Match): ",
          sum([1 if p == r else 0 for p, r in zip(predictions, references)]) / len(predictions))
    print("Accuracy(Levenshtein~Score): ",
          sum([levenshtein_score(p, l) for p, l in zip(predictions, references)]) / len(predictions))


if __name__ == "__main__":
    main()
