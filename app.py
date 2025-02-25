import gradio as gr
from fastai.vision.all import *

__all__ = ['is_irezumi', 'learn', 'categories', 'classify_image']

def is_irezumi(x): 
    return x[0].isupper()

# Ensure the correct file extension for FastAI learner
learn = load_learner('irezumi-v0.0.1.pkl')

categories = ('American', 'Irezumi')

def classify_image(img):
    img = img.resize((192, 192))  # Resize if necessary
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

# Corrected gr.Image component
image = gr.Image(type="pil")  # Ensure it provides a PIL image
label = gr.Label()
examples = ['irezumi.jpg', 'american.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)
