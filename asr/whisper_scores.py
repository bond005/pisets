from typing import Any

import torch
import numpy as np
from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperTokenizerFast,
    WhisperForConditionalGeneration
)

from .comparison import TokenizedText


def whisper_pipeline_transcribe_with_word_scores(
    waveform: np.ndarray,
    recognizer: AutomaticSpeechRecognitionPipeline,
) -> tuple[TokenizedText, list[list[str]], list[list[float]]]:
    """
    A wrapper around `.whisper_transcribe_with_word_scores()` to use a pipeline.
    Example:

    ```
    import librosa
    from asr.asr import initialize_model_for_speech_recognition
    waveform, _ = librosa.load('tests/testdata/test_sound_ru.wav', sr=None)
    pipeline = initialize_model_for_speech_recognition()
    whisper_pipeline_transcribe_with_word_scores(waveform, pipeline)
    ```
    """
    return whisper_transcribe_with_word_scores(
        waveform,
        recognizer.feature_extractor,
        recognizer.tokenizer,
        recognizer.model,
        recognizer._forward_params,  # lang, task
    )


def whisper_transcribe_with_word_scores(
    waveform: np.ndarray,
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer | WhisperTokenizerFast,
    model: WhisperForConditionalGeneration,
    generate_kwargs: dict[str, Any],
) -> tuple[TokenizedText, list[list[str]], list[list[float]]]:
    """
    Transcribes the audio with Whisper and returns:
    - the resulting text tokenized into words
    - a list of tokens for each word
    - a list of token scores for each word

    Example:
    ```
    import librosa
    waveform, _ = librosa.load('tests/testdata/test_sound_ru.wav', sr=None)
    recognizer = pipeline('automatic-speech-recognition', model='openai/whisper-large-v3')
    whisper_transcribe_with_word_scores(
        waveform,
        recognizer.feature_extractor,
        recognizer.tokenizer,
        recognizer.model,
        {'language': '<|ru|>', 'task': 'transcribe'},  # or `recognizer._forward_params`
    )

    >>> (
        TokenizedText(
            text=' нейронные сети это хорошо.',
            tokens=[
                Substring(start=1, stop=10, text='нейронные', is_punct=False),
                Substring(start=11, stop=15, text='сети', is_punct=False),
                Substring(start=16, stop=19, text='это', is_punct=False),
                Substring(start=20, stop=26, text='хорошо', is_punct=False),
                Substring(start=26, stop=27, text='.', is_punct=True)
            ]
        ),
        [[' ней', 'рон', 'ные'], [' с', 'ети'], [' это'], [' хорошо']],
        [[-0.61, -6.80e-05, -0.00], [-8.82e-05, -2.41e-05], [-0.57], [-0.00]]
    )
    ```
    """
    assert model.config.model_type == 'whisper'

    inputs = feature_extractor(
        waveform,
        return_tensors='pt',
        sampling_rate=16_000,
    ).to(model.device, model.dtype)
    result = model.generate(
        **inputs,
        **generate_kwargs,
        return_dict_in_generate=True,
        return_token_timestamps=True,
    )

    # convert token ids and logits to numpy
    token_ids = result['sequences'][0].cpu().numpy()
    logits = torch.nn.functional.log_softmax(torch.stack(result['scores']), dim=-1).cpu().numpy()

    # skip start special tokens to align with logits
    token_ids = token_ids[-len(logits):]

    # skip all special tokens
    is_special = np.array([id in tokenizer.all_special_ids for id in token_ids])
    token_ids = token_ids[~is_special]
    logits = logits[~is_special]

    score_per_token = np.array([float(l[0, token_id]) for token_id, l in zip(token_ids, logits)])

    # reproducing whisper bpe decoding
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    bytes_list_per_token = [
        [byte_decoder[x] for x in bytes_str]
        for bytes_str in tokenizer.convert_ids_to_tokens(token_ids)
    ]

    # searching for token positions in the text
    token_end_positions = []
    for i in range(len(bytes_list_per_token)):
        concatenated_bytes = sum(bytes_list_per_token[:i + 1], [])
        try:
            text = bytearray(concatenated_bytes).decode('utf-8', errors='strict')
            token_end_positions.append(len(text))
        except UnicodeDecodeError:
            token_end_positions.append(None)  # not a full utf-8 charachter

    assert text == tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)

    # cleaning up tokenization spaces, shifting token_end_positions
    # (see .clean_up_tokenization() in PreTrainedTokenizerBase)
    if tokenizer.clean_up_tokenization_spaces:
        for replace_from in [" .", " ?", " !", " ,", " ' ", " n't", " 'm", " 's", " 've", " 're"]:
            replace_to = replace_from.strip()
            while (start_pos := text.find(replace_from)) != -1:
                delta_len = len(replace_to) - len(replace_from)
                text = text[:start_pos] + replace_to + text[start_pos + len(replace_from):]
                token_end_positions = [
                    (
                        token_end_pos
                        if token_end_pos <= start_pos
                        else token_end_pos + delta_len
                    )
                    for token_end_pos in token_end_positions
                ]

        assert text == tokenizer.decode(token_ids)

    # tokenizing the text
    tokenized_text = TokenizedText.from_text(text)

    # matching words and tokens
    tokens_range_per_word = []
    for word in tokenized_text.get_words():
        first_token_idx = None  # first token of the word, inclusive
        for token_idx, token_end_pos in enumerate(token_end_positions):
            if token_end_pos is None:
                continue
            if token_end_pos > word.start and first_token_idx is None:
                first_token_idx = token_idx
            if token_end_pos >= word.stop:
                break
        tokens_range_per_word.append((first_token_idx, token_idx + 1))

    tokens_per_word = [
        [
            bytearray(b).decode('utf-8', errors='replace')
            for b in bytes_list_per_token[start_token_idx:end_token_idx]
        ]
        for start_token_idx, end_token_idx in tokens_range_per_word
    ]

    token_scores_per_word = [
        list(score_per_token[start_token_idx:end_token_idx])
        for start_token_idx, end_token_idx in tokens_range_per_word
    ]

    return tokenized_text, tokens_per_word, token_scores_per_word