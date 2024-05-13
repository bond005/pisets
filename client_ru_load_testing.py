import aiohttp
import asyncio
from datasets import load_dataset

async def transcribe_audio(audio_fname, server_address):
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{server_address}/transcribe', files={'audio': open(audio_fname, 'rb')}) as resp:
            response_json = await resp.json()
            return response_json

async def main():
    server_address = "http://localhost:8040/transcribe"

    dataset = load_dataset("dangrebenkin/voxforge-ru-dataset")
    audio_files = dataset["train"]["path"]

    await asyncio.gather(*[transcribe_audio(audio_fname, server_address) for audio_fname in audio_files])

if __name__ == "__main__":
    asyncio.run(main())
