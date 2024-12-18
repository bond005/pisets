import os
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd
import requests

API_URL = os.getenv("PISETS_API_URL", "http://localhost:8040")
status_columns = ["ID задачи", "Имя файла", "Статус", "Скачать"]


def upload_audio(audio_file_path):
    if audio_file_path is None:
        return "Please upload an audio file."
    audio_file_path = Path(audio_file_path)

    # Отправка аудиофайла на обработку
    files = {'audio': open(audio_file_path, 'rb')}
    response = requests.post(f"{API_URL}/transcribe", files=files)

    if response.status_code == 200:
        task_id = response.json()['task_id']
        return task_id
    else:
        return f"Error: {response.text}"


def get_all_statuses():
    response = requests.get(f"{API_URL}/statuses")

    if response.status_code == 200:
        statuses = response.json()
        # Создаем список словарей для DataFrame
        tasks_list = []
        for task_id, task_info in statuses.items():
            if not isinstance(task_info, dict):
                task_info = task_info.json()
            # Извлекаем статус из ответа API
            status = task_info.get('status', 'Unknown')
            source_filename = task_info.get('source_filename', 'Unknown')

            tasks_list.append({
                status_columns[0]: task_id,
                status_columns[1]: source_filename,
                status_columns[2]: status,
                status_columns[3]: "⬇️" if status == "Ready" else "-"
            })

        # Создаем DataFrame
        df = pd.DataFrame(tasks_list)
        return df
    else:
        return pd.DataFrame(columns=status_columns)


def download_result(task_id, filename):
    if not task_id:
        return None

    response = requests.get(f"{API_URL}/download_result/{task_id}")

    if response.status_code == 200:
        # Сохраняем результат во временный файл
        directory = tempfile.mkdtemp(task_id)
        output_path = Path(directory) / f"{filename}.docx"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return_path = str(output_path.absolute().resolve())
        return return_path
    else:
        return None


def handle_download_click(evt: gr.SelectData, statuses_df):
    # Получаем task_id из выбранной строки
    task_id = statuses_df.iloc[evt.index[0]][status_columns[0]]
    filename = statuses_df.iloc[evt.index[0]][status_columns[1]]
    return download_result(task_id, filename)


with gr.Blocks() as demo:
    gr.Markdown("# Сервис автоматического распознавания речи")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Загрузка файла")
            upload_button = gr.Button("Отправить на обработку")

    with gr.Row():
        with gr.Column():
            status_table = gr.Dataframe(
                headers=status_columns,
                interactive=False,
            )
            download_output = gr.File(label="Скачанные результаты")

    # События
    upload_button.click(
        fn=upload_audio,
        inputs=audio_input,
        outputs=None
    )

    status_table.select(
        fn=handle_download_click,
        inputs=status_table,
        outputs=download_output
    )

    status_table.attach_load_event(get_all_statuses, 1, None, )

demo.launch(server_name="0.0.0.0", server_port=80, ssl_verify=False, debug=False)
