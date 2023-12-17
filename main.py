import requests
import json
import argparse

import gradio as gr
from PIL import Image

from diffusers import AutoPipelineForText2Image
import torch
import torch_directml

from helpers import VERSION2SPECS

# Naver Papago API
CLIENT_ID = None
CLIENT_SECRET = None

dml = torch_directml.device()


def korean_to_english(input_text: str) -> str:
    url = "https://openapi.naver.com/v1/papago/n2mt"
    headers = {
        "Content-Type": "application/json",
        "X-Naver-Client-Id": CLIENT_ID,
        "X-Naver-Client-Secret": CLIENT_SECRET,
    }
    data = {"source": "ko", "target": "en", "text": input_text}

    res = requests.post(url, json.dumps(data), headers=headers)
    if res.status_code != requests.codes.ok:
        raise Exception("request failed")

    return res.json()["message"]["result"]["translatedText"]


def create_image(input_text: str) -> Image:
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to(dml)

    image = pipe(prompt=input_text, num_inference_steps=1, guidance_scale=0.0).images[0]
    print(f"Image size : W {image.width}, H {image.height}")

    return image


def create_video(image: Image):
    version = VERSION2SPECS["svd_xt"]


def text_to_video(input_text: str):
    translated_text = korean_to_english(input_text)
    print(f"result : {translated_text}")
    # created_image = create_image(translated_text)
    return translated_text


with gr.Blocks(theme="soft") as demo:
    with gr.Row():
        with gr.Column(min_width=600):
            kr_text = gr.Textbox(label="input", placeholder="입력해 주세요.")
            kr_submit = gr.Button("Submit")
        with gr.Column(min_width=600):
            en_text = gr.Textbox(label="input", interactive=True)
            en_submit = gr.Button("Submit")
    with gr.Row():
        generated_image = gr.Image(interactive=False)
    with gr.Row():
        generated_video = gr.Video(interactive=False)

    kr_submit.click(text_to_video, inputs=kr_text, outputs=en_text)


if __name__ == "__main__":
    demo.launch()
