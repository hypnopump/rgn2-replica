# Author: Serhiy Shekhovtsov

import random
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
import torch
from dataclasses import dataclass
from typing import Optional, Dict, List, Union


@dataclass
class DataCollatorForSpanPermutationLM:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
        self, batch: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(batch[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(batch, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            raise NotImplementedError('Not implemented')

        # if special token mask has been preprocessed, pop it from the dict
        special_tokens_masks = batch.pop("special_tokens_mask", None)
        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        batch_inputs = batch["input_ids"]
        batch_labels = batch["labels"] = batch_inputs.clone()

        for i, (inputs, labels) in enumerate(zip(batch_inputs, batch_labels)):
            p = random.random()

            if special_tokens_masks is None:
                special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)
            else:
                special_tokens_mask = special_tokens_mask[i]

            # 70% will go through span masking
            if p > 0.3:
                inputs = self.span_masking(inputs, special_tokens_mask, mask_token_id)
            #  35% of the remaining 30% will go through chunk permutation
            if p < 0.35 * 0.3:
                continue  # not working yet
                inputs = self.chunk_permutation(inputs, special_tokens_mask, mask_token_id)

            labels[inputs != mask_token_id] = -100  # we only compute loss on masked tokens

            batch_inputs[i] = inputs
            batch_labels[i] = labels

        return batch

    def span_masking(self, inputs, special_tokens_mask, mask_token_id,
                     num_masks_ratio=.15, span_len_range=(2, 8), single_word_mask_proba=.7):
        """With probability of `single_word_mask_proba` will generate single-token masks.
        Otherwise will generate masks of random size within the range of `span_len_range`.
        Num of masks is `seq_len * num_masks_ratio`
        """
        p = random.random()

        # masking single word instead of span with specified probability
        if p < single_word_mask_proba:
            span_len_range = 1, 1

        num_masks = int(len(inputs) * num_masks_ratio)
        for _ in range(num_masks):
            span_len = random.randint(*span_len_range)
            span_start = random.randint(0, len(inputs) - 1 - span_len)
            span_end = span_start+span_len

            if special_tokens_mask[span_start:span_end].sum() > 0:
                # skip masking if it's interfering with special tokens
                continue

            inputs[span_start:span_end] = mask_token_id

            if (inputs == mask_token_id).sum() > num_masks:
                # we've covered enough of tokens
                break

        return inputs

    def chunk_permutation(self):
        raise NotImplementedError('chunk_permutation is not implemented yet')
