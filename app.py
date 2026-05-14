from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper() 

learn = load_learner('model.pkl')


categories = ('Dog', 'Cat')

def classify_img(img):
    img = PILImage.create(img)
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))



image = gr.Image()
label = gr.Label()
examples = ['dog.jfif', 'cat.jpg', 'random.jpg']

intf = gr.Interface(
    fn=classify_img,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    examples=["dog.jfif", "cat.jpg"]
)

intf.launch()

 