from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
)
from dataclasses import dataclass, field
from typing import Union, List
from utils_generation import GenerationConfig


@dataclass
class ModelParams:
    model_name_or_path: str = field(
        default="psyche/KoT5",
        metadata={ "help": "모델의 (huggingface-hub)이름 또는 로컬 경로를 설정합니다."}
    )


def main():
    parser = HfArgumentParser((ModelParams,GenerationConfig))
    (model_params, gen_params) = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_params.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_params.model_name_or_path)

    def predict(text:Union[str, List[str]]):
        inputs = tokenizer(text, return_tensors="pt")
        output = model.generate(**inputs, generation_config=gen_params)
        return tokenizer.batch_decode(output, skip_special_tokens=True)

    print(predict(
        "안녕하세요. 만나서 반갑습니다. 저는 홍길동 입니다. 가나다대학교를 졸업했습니다. 이름:, 대학교:",
    ))


if __name__ == "__main__":
    main()