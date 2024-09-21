from argparse import ArgumentParser
import requests
import os
import time


CHUNK_SIZE: int = 8_192
OK_STATUS: int = 200


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input sound file name.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output DocX file name.')
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

    if args.address.strip() == '':
        server_address = 'http://localhost:8040'
    else:
        server_address = f'http://{args.address.strip()}:8040'
    print(f'Server address is {server_address}')
    resp = requests.get(server_address + '/ready')
    if resp.status_code != OK_STATUS:
        raise ValueError(f'The service is not available!')
    
    with open(audio_fname, 'rb') as audio_fp:
        files = {'audio': (audio_fname, audio_fp, 'audio/wave')}
        resp = requests.post(server_address + '/transcribe', files=files)

    if resp.status_code != OK_STATUS:
        raise ValueError(f'The file "{audio_fname}" is not transcribed! ' + str(resp))
    task_id = resp.json()["task_id"]
    print(f'Task ID is {task_id}')

    resp = requests.get(server_address + f'/status/{task_id}')
    while resp.status_code != OK_STATUS:
        time.sleep(5)
        resp = requests.get(server_address + f'/status/{task_id}')

    resp = requests.get(server_address + f'/download_result/{task_id}')
    if resp.status_code != OK_STATUS:
        raise ValueError(f'The file "{audio_fname}" is not downloaded! ' + str(resp))
    with open(output_srt_fname, mode='wb') as dst_fp:
        dst_fp.write(resp.content)


if __name__ == '__main__':
    main()
