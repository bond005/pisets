import logging

from gigachat import GigaChat


gigachat_logger = logging.getLogger(__name__)


def build_prompt_for_detalization(source_text: str) -> str:
    prepared_text = ' '.join(source_text.strip().split())
    if len(prepared_text) == 0:
        gigachat_logger.warning('The input text is empty!')
        return ''
    return 'Сочини, пожалуйста, длинный рассказ на следующую тему. ' + prepared_text


def build_prompt_for_simplification(source_text: str) -> str:
    prepared_text = ' '.join(source_text.strip().split())
    if len(prepared_text) == 0:
        gigachat_logger.warning('The input text is empty!')
        return ''
    prompt = 'Упрости и сократи, пожалуйста, следующий текст, выразив основную мысль одним-двумя предложениями. '
    prompt += prepared_text
    return prompt


def generate_answer_with_gigachat(prompt: str, credentials: str) -> str:
    prepared_prompt = ' '.join(prompt.strip().split())
    if len(prepared_prompt) == 0:
        gigachat_logger.warning('The input prompt is empty!')
        return ''
    with GigaChat(credentials=credentials, scope='GIGACHAT_API_PERS', verify_ssl_certs=False) as giga:
        response = giga.chat(prompt.strip())
    return response.choices[0].message.content.strip()
