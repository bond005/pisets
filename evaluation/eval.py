import time
import json
from pathlib import Path
from typing import Callable, Literal
from dataclasses import dataclass

import torch
import numpy as np
from datasets import load_dataset, Audio
from transformers import pipeline, Pipeline, WhisperProcessor

from asr.asr import (
    initialize_model_for_speech_segmentation,
    initialize_model_for_speech_classification,
    initialize_model_for_speech_recognition,
    transcribe
)

class TranscribeWhisperPipeline:
    """
    A Whisper baseline to compare with `TranscribePisets`.
    """
    def __init__(self, predictions_name: str):
        self.predictions_name = predictions_name
        self.whisper_pipeline = pipeline(
            'automatic-speech-recognition',
            model='openai/whisper-large-v3',
            chunk_length_s=20,
            stride_length_s=(4, 2),
            device='cuda:0',
            model_kwargs={'attn_implementation': 'sdpa'},
            # torch_dtype=torch.float16,
            generate_kwargs={
                'language': '<|ru|>',
                'task': 'transcribe',
                'forced_decoder_ids': None
            }
        )
    
    def __call__(self, waveform: np.ndarray) -> dict[str, str]:
        return self.whisper_pipeline(waveform)['text']


class TranscribeWhisperLongform(TranscribeWhisperPipeline):
    """
    A Whisper longform baseline to compare with `TranscribePisets`.
    """
    def __init__(self, predictions_name: str, condition_on_prev_tokens: bool):
        super().__init__(predictions_name)
        self.whisper_processor = WhisperProcessor.from_pretrained(
            'openai/whisper-large-v3',
            language='Russian',
            task='transcribe',
        )
        self.condition_on_prev_tokens = condition_on_prev_tokens
    
    def __call__(self, waveform: np.ndarray) -> dict[str, str]:
        # https://github.com/huggingface/transformers/pull/27658
        inputs = self.whisper_processor(
            waveform,
            return_tensors='pt',
            truncation=False,
            padding='longest',
            return_attention_mask=True,  # probably we do not need this for Whisper
            sampling_rate=16_000
        )
        result = self.whisper_pipeline.model.generate(
            **inputs.to('cuda'),
            condition_on_prev_tokens=self.condition_on_prev_tokens,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            logprob_threshold=-1.0,
            compression_ratio_threshold=1.35,
            return_timestamps=True,
            language='<|ru|>',
            task='transcribe',
        )
        return self.whisper_processor.batch_decode(result, skip_special_tokens=True)[0]


@dataclass
class TranscribePisets:
    """
    A Pisets wrapper for evaluation purposes.
    
    Transcribes waveform with Pisets and returns results for all stages.

    In contrast to asr.asr.transcribe() this class:
    - Concatenates transcriptions for all segments
    - Does not return timestamps
    - Allows to define custom names for all stages
    """
    
    segmenter: Pipeline | Callable
    vad: Pipeline | Callable | Literal['skip']
    asr: Pipeline | Callable | Literal['skip']

    min_segment_size: int = 1
    max_segment_size: int = 20
    stretch: tuple[int, int] | None = None

    segmenter_predictions_name: str | None = None
    asr_predictions_name: str | None = None
    asr_stretched_predictions_name: str | None = None
    
    def __call__(self, waveform: np.ndarray) -> dict[str, str]:
        # transcribing
        outputs = transcribe(
            waveform,
            segmenter=self.segmenter,
            voice_activity_detector=(
                self.vad
                if self.vad != 'skip'
                else (lambda audio: [{'score': 1, 'label': 'Speech'}])
            ),
            asr=(
                self.asr
                if self.asr != 'skip'
                else (lambda audio: {'text': ''})
            ),
            min_segment_size=self.min_segment_size,
            max_segment_size=self.max_segment_size,
            stretch=self.stretch,
        )
        # concatenating segments
        results = {}
        if self.segmenter_predictions_name is not None:
            results[self.segmenter_predictions_name] = ' '.join([s.transcription_from_segmenter for s in outputs])
        if self.asr_predictions_name is not None:
            results[self.asr_predictions_name] = ' '.join([s.transcription for s in outputs])
        if self.asr_stretched_predictions_name is not None:
            results[self.asr_stretched_predictions_name] = ' '.join([s.transcription_stretched for s in outputs])
        return results
    
# defining transcribers without instantiating them all at once to save GPU memory

transcribers = {
    'Whisper pipeline': lambda: TranscribeWhisperPipeline(
        predictions_name='Baseline Whisper pipeline',
    ),
    'Whisper longform': lambda: TranscribeWhisperLongform(
        predictions_name='Baseline Whisper longform',
        condition_on_prev_tokens=False,
    ),
    'Whisper longform conditioned': lambda: TranscribeWhisperLongform(
        predictions_name='Baseline Whisper longform conditioned',
        condition_on_prev_tokens=True,
    ),
    'Pisets (segments 1s-20s)': lambda: TranscribePisets(
        segmenter=initialize_model_for_speech_segmentation('ru', 'bond005/wav2vec2-large-ru-golos-with-lm'),
        vad=initialize_model_for_speech_classification(),
        asr=initialize_model_for_speech_recognition('ru', 'openai/whisper-large-v3'),
        min_segment_size=1,
        max_segment_size=20,
        stretch=(3, 4),
        segmenter_predictions_name='W2V2 Golos LM',
        asr_predictions_name='Pisets WhisperV3 (segments 1s-20s)',
        asr_stretched_predictions_name='Pisets WhisperV3 stretched (segments 1s-20s)',
    ),
    'Pisets (segments 10s-30s)': lambda: TranscribePisets(
        segmenter=initialize_model_for_speech_segmentation('ru', 'bond005/wav2vec2-large-ru-golos-with-lm'),
        vad=initialize_model_for_speech_classification(),
        asr=initialize_model_for_speech_recognition('ru', 'openai/whisper-large-v3'),
        min_segment_size=10,
        max_segment_size=30,
        asr_predictions_name='Pisets WhisperV3 (segments 10s-30s)',
    ),
    'W2V2 golos no LM': lambda: TranscribePisets(
        segmenter=initialize_model_for_speech_segmentation('ru', 'bond005/wav2vec2-large-ru-golos'),
        vad='skip',
        asr='skip',
        segmenter_predictions_name='W2V2 Golos no LM',
    ),
    'Pisets Podlodka': lambda: TranscribePisets(
        segmenter=initialize_model_for_speech_segmentation('ru', 'bond005/wav2vec2-large-ru-golos-with-lm'),
        vad=initialize_model_for_speech_classification(),
        asr=initialize_model_for_speech_recognition('ru', 'bond005/whisper-large-v3-ru-podlodka'),
        min_segment_size=1,
        max_segment_size=20,
        asr_predictions_name='Pisets WhisperV3 Podlodka (segments 1s-20s)',
    ),
    'Pisets no-VAD': lambda: TranscribePisets(
        segmenter=initialize_model_for_speech_segmentation('ru', 'bond005/wav2vec2-large-ru-golos-with-lm'),
        vad='skip',
        asr=initialize_model_for_speech_recognition('ru', 'openai/whisper-large-v3'),
        min_segment_size=1,
        max_segment_size=20,
        asr_predictions_name='Pisets WhisperV3 no-VAD (segments 1s-20s)',
    ),
    'Pisets no-VAD Podlodka': lambda: TranscribePisets(
        segmenter=initialize_model_for_speech_segmentation('ru', 'bond005/wav2vec2-large-ru-golos-with-lm'),
        vad='skip',
        asr=initialize_model_for_speech_recognition('ru', 'bond005/whisper-large-v3-ru-podlodka'),
        min_segment_size=1,
        max_segment_size=20,
        asr_predictions_name='Pisets WhisperV3 no-VAD Podlodka (segments 1s-20s)',
    ),
}

dataset = (
    load_dataset('dangrebenkin/long_audio_youtube_lectures')
    .cast_column('audio', Audio(sampling_rate=16_000))
    ['train']
)

output_dir = Path('/home/oleg/pisets_test_results')
output_dir.mkdir(parents=True, exist_ok=True)

for transcriber_name, transcriber_lambda in transcribers.items():

    # instantiate transcriber on GPU
    transcriber = transcriber_lambda()

    for sample in dataset:
        print(filepath := output_dir / f'{sample["name"]} {transcriber_name}.json')

        torch.cuda.reset_peak_memory_stats()

        if filepath.is_file():
            print(f'Already exists')
            continue

        start_time = time.time()
        transcriptions = transcriber(sample['audio']['array'])
        print('Elapsed', elapsed_time := time.time() - start_time)

        with open(filepath, 'w') as f:
            json.dump({
                'audio_name': sample['name'],
                'transcriber_name': transcriber_name,
                'elapsed_time': elapsed_time,
                'transcriptions': transcriptions,
            }, f)

        print(f'GPU max allocated memory: {torch.cuda.max_memory_allocated(0) / 2**30:.2f} GB')