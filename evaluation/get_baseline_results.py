from pathlib import Path
from transformers import pipeline, Pipeline, WhisperProcessor, WhisperForConditionalGeneration
import pysrt
import librosa

from asr.comparison import MultipleTextsAlignment

recognizer = pipeline(
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
whisper_processor = WhisperProcessor.from_pretrained(
    'openai/whisper-large-v3',
    language='Russian',
    task='transcribe',
)

def pipeline_transcribe_with_whisper(
    waveform: str,
    pipeline: Pipeline,
) -> str:
    return pipeline(waveform)['text']

def longform_transcribe_with_whisper(
    waveform: str,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    condition_on_prev_tokens: bool = False,
) -> str:
    # https://github.com/huggingface/transformers/pull/27658
    inputs = processor(
        waveform,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
        sampling_rate=16_000
    ).to("cuda")
    result = model.generate(
        **inputs,
        condition_on_prev_tokens=condition_on_prev_tokens,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        logprob_threshold=-1.0,
        compression_ratio_threshold=1.35,
        return_timestamps=True,
        language='<|ru|>',
        task='transcribe',
    )
    return whisper_processor.batch_decode(result, skip_special_tokens=True)[0]

input_dir = Path('/home/oleg/pisets_test_set/')
output_dir = Path('/home/oleg/pisets_test_results/')

for audio_path in input_dir.glob('*.wav'):

    if (srt_path := audio_path.with_suffix('.srt')).is_file():
        truth = ' '.join([sub.text for sub in pysrt.open(srt_path)])
    else:
        truth = open(audio_path.with_suffix('.txt')).read()
    
    long_waveform, _ = librosa.load(audio_path, sr=16_000)
    print(f'{audio_path.stem} {len(long_waveform) / 16_000} sec')

    pred = pipeline_transcribe_with_whisper(long_waveform, recognizer)
    print('pipeline', MultipleTextsAlignment.from_strings(truth, pred).wer())
    with open(output_dir / f'{audio_path.stem}_only_whisper_pipeline.txt', 'w') as f:
        f.write(pred)

    pred = longform_transcribe_with_whisper(long_waveform, whisper_processor, recognizer.model)
    print('longform', MultipleTextsAlignment.from_strings(truth, pred).wer())
    with open(output_dir / f'{audio_path.stem}_only_whisper_longform.txt', 'w') as f:
        f.write(pred)

    pred = longform_transcribe_with_whisper(long_waveform, whisper_processor, recognizer.model, condition_on_prev_tokens=True)
    print('longform conditioned', MultipleTextsAlignment.from_strings(truth, pred).wer())
    with open(output_dir / f'{audio_path.stem}_only_whisper_longform_conditioned.txt', 'w') as f:
        f.write(pred)