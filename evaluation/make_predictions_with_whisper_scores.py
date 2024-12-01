import json
from pathlib import Path
import dataclasses

from datasets import load_dataset, Audio
from tqdm.auto import tqdm

from asr.asr import (
    initialize_model_for_speech_segmentation,
    initialize_model_for_speech_recognition,
    transcribe
)
from asr.comparison import TokenizedText
from asr.whisper_scores import whisper_pipeline_transcribe_with_word_scores


dataset = (
    load_dataset('dangrebenkin/long_audio_youtube_lectures')
    .cast_column('audio', Audio(sampling_rate=16_000))
    ['train']
)

output_dir = Path('/home/oleg/pisets_test_results_with_scores')
output_dir.mkdir(parents=True, exist_ok=True)

segmenter = initialize_model_for_speech_segmentation('ru', 'bond005/wav2vec2-large-ru-golos')
whisper_pipeline = initialize_model_for_speech_recognition('ru', 'openai/whisper-large-v3')

for sample in dataset:
    print(sample['name'])

    waveform = sample['audio']['array']

    results = transcribe(
        waveform,
        segmenter=segmenter,
        voice_activity_detector=lambda audio: [{'score': 1, 'label': 'Speech'}],
        asr=lambda audio: {'text': 'none'},
        min_segment_size=1,
        max_segment_size=20,
    )

    tokenized_segments = []
    scores_per_word = []

    for segment in tqdm(results, desc='whisper'):
        waveform_segment = waveform[int(segment.start * 16_000):int(segment.end * 16_000)]
        tokenized_text_for_segment, _, scores_for_segment = (
            whisper_pipeline_transcribe_with_word_scores(waveform_segment, whisper_pipeline)
        )
        tokenized_segments.append(tokenized_text_for_segment)
        scores_per_word += scores_for_segment

    tokenized_text = TokenizedText.concatenate(tokenized_segments)

    transcriber_name = 'Pisets WhisperV3 no-VAD (segments 1s-20s) with scores'

    filepath = output_dir / f'{sample["name"]} {transcriber_name}.json'

    with open(filepath, 'w') as f:
        json.dump({
            'audio_name': sample['name'],
            'transcriber_name': transcriber_name,
            'tokenized_text': dataclasses.asdict(tokenized_text),
            'scores_per_word': scores_per_word,
        }, f, ensure_ascii=False)