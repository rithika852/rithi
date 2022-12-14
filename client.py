import gradio as gr
from model import Model


model = Model(model_path="pretrained_model")

def process_file(file):
    file_path = str(file)
    return model.predict(file_path) 

video_file = gr.Video()
audio_text = gr.TextArea()

demo = gr.Interface(fn=process_file,
                    inputs=[video_file],
                    outputs=[audio_text])

demo.launch(inline=True, debug=True)