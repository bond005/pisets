from typing import Literal

import numpy as np

import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerBase
from transformers.generation.utils import GenerationMixin

class SequenceScore:
    """
    Calculates a sequence score for a text from an autoregressive LM.
    """
    def __init__(
        self,
        name: str | None = 'ai-forever/rugpt3large_based_on_gpt2',
        tokenizer: PreTrainedTokenizerBase | None = None,
        model: GenerationMixin | None = None,
    ):
        if name is not None:
            assert not tokenizer and not model
            # https://stackoverflow.com/a/75242984
            tokenizer = AutoTokenizer.from_pretrained(name, add_bos_token=True)
            model = AutoModelForCausalLM.from_pretrained(name)
        else:
            assert tokenizer and model


        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, text: str) -> int:
        inputs = self.tokenizer([text], return_tensors='pt')
        logits = self.model(**inputs, return_dict=True).logits[:, :-1]
        targets = inputs['input_ids'][:, 1:]
        logloss = F.cross_entropy(input=logits.transpose(1, 2), target=targets)
        logloss = logloss.cpu().detach().numpy()

        if np.isnan(logloss):
            return 0 # TODO why happens?

        return -logloss