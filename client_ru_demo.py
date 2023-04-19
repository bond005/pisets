from argparse import ArgumentParser
import codecs
import requests
import os


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input sound file name or YouTube URL.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output SubRip file name.')
    parser.add_argument('-a', '--address', dest='address', type=str, required=False, default='',
                        help='The web address of the service (if it is not a local).')
    args = parser.parse_args()

    audio_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(audio_fname):
        err_msg = f'The file "{audio_fname}" does not exist!'
        raise IOError(err_msg)

    output_srt_fname = os.path.normpath(args.output_name)
    output_srt_dir = os.path.dirname(output_srt_fname)
    if len(output_srt_dir) > 0:
        if not os.path.isdir(output_srt_dir):
            err_msg = f'The directory "{output_srt_dir}" does not exist!'
            raise IOError(err_msg)
    if len(os.path.basename(output_srt_fname).strip()) == 0:
        err_msg = f'The file name "{output_srt_fname}" is incorrect!'
        raise IOError(err_msg)

    resp = requests.get('http://localhost:8040/ready')
    if resp.status_code != 200:
        raise ValueError(f'The service is not available!')
    
    with open(audio_fname, 'rb') as audio_fp:
        files = {'audio': (audio_fname, audio_fp, 'audio/wave')}
        if args.address.strip() != '':
            resp = requests.post('http://' + args.address.strip() + ':8040/transcribe', files=files)
        else:
            resp = requests.post('http://localhost:8040/transcribe', files=files)

    if resp.status_code != 200:
        raise ValueError(f'The file "{audio_fname}" is not transcribed! ' + str(resp))
    with codecs.open(output_srt_fname, mode='w', encoding='utf-8') as srt_fp:
        srt_fp.write(resp.json())


if __name__ == '__main__':
    main()
