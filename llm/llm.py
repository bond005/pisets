import logging
from typing import Optional, Tuple

import torch
from peft import PeftModel, PeftConfig
from transformers import MistralForCausalLM, LlamaTokenizer, GenerationConfig


saiga_mistral_logger = logging.getLogger(__name__)
SAIGA_MISTRAL_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>"
SAIGA_MISTRAL_RESPONSE_TEMPLATE = "<s>bot\n"
SAIGA_MISTRAL_SYSTEM_PROMPT = ("Ты — Сайга, русскоязычный автоматический ассистент. "
                               "Ты разговариваешь с людьми и помогаешь им.")


class Conversation:
    def __init__(
        self,
        message_template=SAIGA_MISTRAL_MESSAGE_TEMPLATE,
        system_prompt=SAIGA_MISTRAL_SYSTEM_PROMPT,
        response_template=SAIGA_MISTRAL_RESPONSE_TEMPLATE
    ):
        self.message_template = message_template
        self.response_template = response_template
        self.messages = [{
            'role': 'system',
            'content': system_prompt
        }]

    def add_user_message(self, message):
        self.messages.append({
            'role': 'user',
            'content': message
        })

    def add_bot_message(self, message):
        self.messages.append({
            'role': 'bot',
            'content': message
        })

    def get_prompt(self) -> str:
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += SAIGA_MISTRAL_RESPONSE_TEMPLATE
        return final_text.strip()


def initialize_saiga_mistral(model_path: str, base_model_path: Optional[str] = None) -> (
        Tuple)[LlamaTokenizer, PeftModel, GenerationConfig]:
    tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
    saiga_mistral_logger.info(f'The tokenizer is loaded from the "{model_path}".')
    config = PeftConfig.from_pretrained(model_path)
    model = MistralForCausalLM.from_pretrained(
        config.base_model_name_or_path if base_model_path is None else base_model_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map='cuda:0' if torch.cuda.is_available() else 'auto'
    )
    info_msg = (f'The base model is loaded from the '
                f'"{config.base_model_name_or_path if base_model_path is None else base_model_path}".')
    saiga_mistral_logger.info(info_msg)
    model = PeftModel.from_pretrained(
        model,
        model_path,
        torch_dtype=torch.float16
    )
    model.eval()
    saiga_mistral_logger.info(f'The adapter is loaded from the "{model_path}".')
    generation_config = GenerationConfig.from_pretrained(model_path)
    saiga_mistral_logger.info(f'The generation config is loaded from the "{model_path}".')
    return tokenizer, model, generation_config


def build_prompt_for_detalization(source_text: str) -> str:
    prepared_text = ' '.join(source_text.strip().split())
    if len(prepared_text) == 0:
        saiga_mistral_logger.warning('The input text is empty!')
        return ''
    return 'Сочини, пожалуйста, длинный рассказ на следующую тему. ' + prepared_text


def build_prompt_for_simplification(source_text: str) -> str:
    prepared_text = ' '.join(source_text.strip().split())
    if len(prepared_text) == 0:
        saiga_mistral_logger.warning('The input text is empty!')
        return ''
    prompt = 'Упрости и сократи, пожалуйста, следующий текст, выразив основную мысль одним-двумя предложениями. '
    prompt += prepared_text
    return prompt


def generate_answer_with_saiga_mistral(prompt: str, tokenizer: LlamaTokenizer, model: PeftModel,
                                       generation: GenerationConfig) -> str:
    prepared_prompt = ' '.join(prompt.strip().split())
    if len(prepared_prompt) == 0:
        saiga_mistral_logger.warning('The input prompt is empty!')
        return ''
    conversation = Conversation()
    conversation.add_user_message(prompt.strip())
    full_prompt = conversation.get_prompt()
    data = tokenizer(full_prompt, return_tensors='pt', add_special_tokens=False)
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation
    )[0]
    output_ids = output_ids[len(data['input_ids'][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()
