from flask import Flask, render_template, request
from io import BytesIO
import base64

import customtkinter as ctk
from PIL import Image, ImageTk
import torch
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

# Hugging Face authentication token
auth_token = "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"

# Initialize the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
stable_diffusion_model = StableDiffusionPipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
stable_diffusion_model.to(device)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt_text = request.form['prompt']
        output = stable_diffusion_model(prompt_text, guidance_scale=8.5)

        if "images" in output:
            images = output["images"]
            if images:
                image = images[0]
                img_io = BytesIO()
                image.save(img_io, 'PNG')
                img_io.seek(0)
                img_data = base64.b64encode(img_io.getvalue()).decode()

                return render_template('index.html', image_data=img_data)

    return render_template('index.html', image_data=None)


if __name__ == '__main__':
    app.run(debug=True)
