import torch
from transformers import BitsAndBytesConfig
from transformers import pipeline
import requests
from flask import Flask
from PIL import Image

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "llava-hf/llava-1.5-7b-hf"

pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})


app = Flask(__name__)

@app.route("/")
def home():
  return "Success!"

@app.route('/process_image', methods=['GET', 'POST'])
def process_image():
    image_url = "https://llava-vl.github.io/static/images/view.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)

    max_new_tokens = 200
    prompt = "USER: <image>\nWhat are the things I should be cautious about when I visit this place?\nASSISTANT:"

    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

    # text = "Hello, World!"

    # Return the text response
    return outputs[0]["generated_text"]

if __name__ == '__main__':
    app.run()