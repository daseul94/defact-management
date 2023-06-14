from collections import defaultdict
from typing import Optional, Dict, List
import datasets
import evaluate
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    is_datasets_available,
    logging,
)
from transformers.trainer_pt_utils import IterableDatasetShard

from utils_generation import GenerationConfig

import re

def process_text(text: str) -> str:
    text = re.sub("<[^>]+>", "", text)
    text = re.sub("\n\n{2,}", "\n\n", text)
    text = text.strip()
    return text


logger = logging.get_logger(__name__)


def extract_answer_from_text(texts, delimiter=" // ", end_mark="###"):
    texts = [text.split(end_mark)[0] for text in texts]
    texts = [[l for l in text.split(delimiter) if l] for text in texts]
    texts = [process_text(text[-1]) if text else "###" for text in texts]
    texts = ["###" if len(text.strip()) == 0 else text for text in texts]
    return texts


class GenerationTrainer(Trainer):
    def __init__(self, print_generation: bool = False, generation_config=None, **kwargs):
        super().__init__(**kwargs)
        self.print_generation = print_generation
        if generation_config is None:
            generation_config = GenerationConfig()
        self.generation_config = generation_config

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None,
                            data_collator: Optional[DataLoader] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator if data_collator is None else data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        _eval_collator = DataCollatorWithPadding(self.tokenizer)
        eval_loader = self.get_eval_dataloader(eval_dataset, data_collator=_eval_collator)
        _metric_bleu = evaluate.load("bleu")
        _metric_rouge = evaluate.load("rouge")

        total_score = defaultdict(list)
        logger.info("*** Evaluate ***")
        self.model.eval()
        with torch.no_grad():
            for inputs in tqdm(eval_loader):
                inputs = self._prepare_inputs(inputs)
                outputs = self.model.generate(
                    inputs["input_ids"],
                    generation_config=self.generation_config,
                )

                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                inputs["target"] = self.tokenizer.batch_decode(inputs["labels"], skip_special_tokens=True)

                inputs["generated"] = extract_answer_from_text(decoded)
                inputs["target"] = extract_answer_from_text(inputs["target"])

                result_bleu = _metric_bleu.compute(predictions=inputs["generated"],
                                                   references=[[inp] for inp in inputs["target"]])

                result_rouge = _metric_rouge.compute(predictions=inputs["generated"],
                                                     references=[inp for inp in inputs["target"]])

                result_bleu.update({f"bleu-{i + 1}": v for i, v in enumerate(result_bleu["precisions"])})
                result_bleu.update(result_rouge)

                if self.print_generation:
                    print("===== Generated Batch =====")
                    for g, t, og in zip(inputs["generated"], inputs["target"], decoded):
                        print("#" * 50)
                        print(f"Generated(original):{og}\n\nGenerated:\n{g}, \n\nTarget:\n{t}")

                for k, v in result_bleu.items():
                    if k != "precisions":
                        total_score[k].append(v)

        total_score = {k: sum(v) / len(v) for k, v in total_score.items()}

        self.log(total_score)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, total_score)
        self._memory_tracker.stop_and_update_metrics(total_score)
        return total_score
