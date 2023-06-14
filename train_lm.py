from transformers import HfArgumentParser, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from dataclasses import dataclass, field
from datasets import load_dataset
from collections import OrderedDict
from utils_trainer import GenerationTrainer
from utils_generation import GenerationConfig

@dataclass
class ModelParams:
    model_name_or_path: str = field(
        default="KETI-AIR/ke-t5-base",
        metadata={ "help": "모델의 (huggingface-hub)이름 또는 로컬 경로를 설정합니다."}
    )

    model_auth_token: str = field(
        default="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", metadata={"help": "비공개 모델 로드를 위한 토큰을 설정합니다."}
    )

    max_source_length: int = field(
        default=512,
        metadata={"help": "모델의 입력으로 들어갈 최대 길이를 설정합니다."}
    )

    max_target_length: int = field(
        default=256,
        metadata={"help": "모델의 출력으로 들어갈 최대 길이를 설정합니다."}
    )

    print_generation: bool = field(
        default=True,
        metadata={"help": "모델의 Output을 출력할지 여부를 설정합니다."}
    )


@dataclass
class DataParams:
    data_name_or_path: str = field(
        default="klue.wos",
        metadata={"help": "학습 데이터에 대한 정보가 기술되어 있는 json 파일의 경로입니다(data_info.json을 참고해주세요)."}
    )

    data_auth_token: str = field(
        default="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",
        metadata={"help": "비공개 데이터셋을 다운로드하기 위한 토큰을 설정합니다."}
    )

    prefix: str = field(
        default="질문 답변",
        metadata={"help": "데이터셋의 prefix를 설정합니다."}
    )

    text_column: str = field(
        default="context",
        metadata={"help": "데이터셋의 텍스트 컬럼의 이름을 설정합니다."}
    )

    text_pair_column: str = field(
        default="prompt",
        metadata={"help": "데이터셋의 텍스트 컬럼의 이름을 설정합니다."}
    )

    label_column: str = field(
        default="answer",
        metadata={"help": "데이터셋의 라벨 컬럼의 이름을 설정합니다."}
    )

    train_split: str = field(
        default="train",
        metadata={"help": "데이터셋의 학습 데이터를 나타내는 split 이름을 설정합니다."}
    )

    eval_split: str = field(
        default="validation",
        metadata={"help": "데이터셋의 검증 데이터를 나타내는 split 이름을 설정합니다."}
    )


def klue_wos_process(examples):
    from collections import defaultdict

    outputs = defaultdict(list)

    for example in examples['dialogue']:
        contexts, labels = [[]], [[]]
        for utter in example:
            if utter['role'] == 'sys':
                contexts.append(contexts[-1].copy())
                labels.append([])
            contexts[-1].append(utter['text'])
            if utter['state']:
                labels[-1].append(utter['state'])

        contexts = [c for c, l in zip(contexts, labels) if l]
        labels = [l[0] for l in labels if l]
        contexts = [" ".join(c) for c in contexts]
        prompts = [", ".join([" ".join(l.split("-")[:-1]) for l in label]) for label in labels]

        labels = [", ".join([" ".join(l.split("-")[:-1])+": "+l.split("-")[-1] for l in label]) for label in labels]
        for c, p, l in zip(contexts, prompts, labels):
            outputs['context'].append(c)
            outputs['prompt'].append(p)
            outputs['answer'].append(l)
    return outputs


_PROCESS_FUNCTION = OrderedDict([
    ("klue.wos", klue_wos_process)
])


def main():
    parser = HfArgumentParser((DataParams, ModelParams, Seq2SeqTrainingArguments, GenerationConfig))
    data_params, model_params, training_params, generation_config = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_params.model_name_or_path, use_auth_token=model_params.model_auth_token, sep_token="<sep>")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_params.model_name_or_path, use_auth_token=model_params.model_auth_token)

    _process_function = _PROCESS_FUNCTION.get(data_params.data_name_or_path, lambda x: x)
    dataset = load_dataset(*data_params.data_name_or_path.split("."), use_auth_token=data_params.data_auth_token)
    dataset = dataset.map(_process_function, batched=True)
    tokenizer.truncation_side = 'left'
    def example_function(examples):
        tokenized_inputs = tokenizer(
            examples[data_params.text_column],
            examples[data_params.text_pair_column],
            padding="max_length",
            truncation=True,
            max_length=model_params.max_source_length,
            return_tensors="pt"
        )

        tokenized_inputs["labels"] = tokenizer(
            examples[data_params.label_column],
            padding="max_length",
            truncation=True,
            max_length=model_params.max_target_length,
            return_tensors="pt"
        )["input_ids"]

        return tokenized_inputs

    dataset = dataset.map(example_function, batched=True, remove_columns=dataset[data_params.train_split].column_names)

    _sample_generator = iter(dataset[data_params.train_split])
    for i in range(3):
        sample = next(_sample_generator)
        print(f"***** Example {i} *****")
        print("[Sample] Input Text:", tokenizer.decode(sample["input_ids"], skip_special_tokens=False))
        print("[Sample] Label Text:", tokenizer.decode(sample["labels"], skip_special_tokens=False))

    dataset = dataset.shuffle(seed=42)
    trainer = GenerationTrainer(
        model=model,
        args=training_params,
        train_dataset=dataset[data_params.train_split],
        eval_dataset=dataset[data_params.eval_split],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        print_generation=model_params.print_generation,
        generation_config=generation_config
    )
    if training_params.do_train:
        trainer.train()
    elif training_params.do_eval:
        result = trainer.evaluate()
        print(result)




if __name__ == "__main__":
    main()