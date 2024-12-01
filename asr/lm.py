from itertools import combinations
from typing import Any

from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerBase
from transformers.generation.utils import GenerationMixin

from asr.comparison import MultipleTextsAlignment


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
        self.model.eval()

    def __call__(self, text: str) -> int:
        inputs = self.tokenizer([text], return_tensors='pt')
        with torch.no_grad():
            logits = self.model(**inputs, return_dict=True).logits[:, :-1]
            targets = inputs['input_ids'][:, 1:]
            logloss = F.cross_entropy(input=logits.transpose(1, 2), target=targets)

        logloss = logloss.cpu().detach().numpy()

        if np.isnan(logloss):
            return 0 # TODO why happens?

        return -logloss


def get_all_subsets(elements: list[Any]):
    """
    Returns all subsets of a list.
    ```
    get_all_subsets([1, 2, 3])
    >>> [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    ```
    """
    return sum((
        [list(x) for x in combinations(elements, r)]
        for r in range(len(elements) + 1)
    ), [])

scorer = SequenceScore('ai-forever/rugpt3large_based_on_gpt2')

def accept_suggestions_by_lm(
    base_vs_additional: MultipleTextsAlignment,
    suggestion_indices: list[int],
    scorer: SequenceScore,
    look_forward: int = 2,
    context_before: int = 100,
    context_after: int = 50,
    pbar: bool = True,
    verbose: bool = False,
) -> list[int]:
    """
    When two predictions disagree, selects one that LM prefers. Returns suggestion_indices
    when the second prediction (`base_vs_additional.text2`) was selected.

    TODO better docstring
    """

    orig_indices_to_resolve = suggestion_indices
    indices_to_resolve = orig_indices_to_resolve.copy()
    indices_accepted = []

    if pbar:
        _pbar = tqdm(total=len(indices_to_resolve))

    while len(indices_to_resolve):
        indices = indices_to_resolve[:look_forward]

        scores = {}

        for indices_to_consider in get_all_subsets(indices):
            text = base_vs_additional.substitute(replace=indices_accepted + indices_to_consider)

            start_idx = base_vs_additional.matches[indices[0]].char_start1
            end_idx = (
                base_vs_additional.matches[indices[-1]].char_end1
                + len(text) - len(base_vs_additional.text1.text)
            )

            start_idx -= context_before
            end_idx += context_after

            start_idx = np.clip(start_idx, 0, len(text))
            end_idx = np.clip(end_idx, 0, len(text))

            text = text[start_idx:end_idx]

            scores[tuple(indices_to_consider)] = {
                'score': scorer(text),
                # 'text' : text
            }

        best_option = max(scores, key=lambda k: scores[k]['score'])

        if verbose:
            print(f'[{len(indices_to_resolve)}] selected {best_option} from {scores}')

        should_accept_index = indices[0] in best_option

        if should_accept_index:
            indices_accepted.append(indices[0])
        
        indices_to_resolve = indices_to_resolve[1:]

        if pbar:
            _pbar.update(1)

    if pbar:
        _pbar.close()
    
    return indices_accepted