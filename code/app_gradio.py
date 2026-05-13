import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

print("Loading model...")
model = tf.keras.models.load_model('best_cnn.h5')
print("Model loaded!")

def predict_digit(image):
    if image is None:
        return "请绘制一个数字"
    
    img = Image.fromarray(image).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    confidence = float(prediction[0][digit]) * 100
    
    return f"识别结果: {digit} (置信度: {confidence:.2f}%)"

with gr.Blocks(title="MNIST 数字识别") as demo:
    gr.Markdown("# 🔢 MNIST 手写数字识别")
    gr.Markdown("在下方画布上绘制一个数字 (0-9)")
    
    with gr.Row():
        sketchpad = gr.Sketchpad(label="绘制区域", shape=(280, 280))
        output = gr.Textbox(label="识别结果", placeholder="识别结果将显示在这里...")
    
    btn_predict = gr.Button("识别数字", variant="primary")
    btn_clear = gr.Button("清除画布")
    
    btn_predict.click(predict_digit, inputs=sketchpad, outputs=output)
    btn_clear.click(lambda: None, None, sketchpad, queue=False)

if __name__ == "__main__":
    demo.launch()