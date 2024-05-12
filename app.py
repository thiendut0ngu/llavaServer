import torch
from transformers import BitsAndBytesConfig
from transformers import pipeline
import requests
from flask import Flask, request
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
    #image_url = "https://llava-vl.github.io/static/images/view.jpg"
    #image = Image.open(requests.get(image_url, stream=True).raw)
    
    result = ""

    image = request.get_json()
    
    pixel_data = image['pixel_data']
    width = image['width']
    height = image['height']
    
    tuple_colors = [tuple(color) for color in pixel_data]
    
    new_image = Image.new('RGB', (width, height))
    new_image.putdata(tuple_colors)
    
    
    # Save the new image as a JPEG file
    new_image.save('reversed_image.jpg', 'JPEG')
    
    image_location = "reversed_image.jpg"
    image = Image.open(image_location)

    #weather
    max_new_tokens = 200
    prompt = "USER: <image>\nWhat is the weather type?\nASSISTANT:"

    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    index = outputs[0]["generated_text"].index("ASSISTANT:")
    result = result + "Weather type: " + outputs[0]["generated_text"][index + len("ASSISTANT:"):].strip() + '\n'
    
    #Road type
    prompt = "USER: <image>\nWhat is the road type?\nASSISTANT:"

    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    index = outputs[0]["generated_text"].index("ASSISTANT:")
    result = result + "Road type: " + outputs[0]["generated_text"][index + len("ASSISTANT:"):].strip() + '\n'
    
    #Road condition (good or damaged road)
    prompt = "USER: <image>\nWhat is the road condition(good or damaged road)?\nASSISTANT:"

    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    index = outputs[0]["generated_text"].index("ASSISTANT:")
    result = result + "Road condition: " + outputs[0]["generated_text"][index + len("ASSISTANT:"):].strip() + '\n'

    #Road condition (good or damaged road)
    prompt = "USER: <image>\nWhat is the road condition(good or damaged road)?\nASSISTANT:"

    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    index = outputs[0]["generated_text"].index("ASSISTANT:")
    result = result + "Road condition: " + outputs[0]["generated_text"][index + len("ASSISTANT:"):].strip() + '\n'

    # Return the text response
    max_new_tokens = 2000
    prompt = "USER: <image>\nDescribe in detail each means of transport that caused the accident (color, quantity, ...)\nASSISTANT:"
    
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": max_new_tokens})
    
    index = outputs[0]["generated_text"].index("ASSISTANT:")
    result = result + "In detail each means of transport that caused the accident (color, quantity, ...): " + outputs[0]["generated_text"][index + len("ASSISTANT:"):].strip()
    
    return result

if __name__ == '__main__':
    app.run(host='localhost', port=5050, debug=True, threaded=True)
