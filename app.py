import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import io
import urllib.request
import os
import onnxruntime as ort

# ── Download MNIST ONNX model on first run ────────────────────────────────────
MODEL_PATH = "mnist.onnx"
MODEL_URL  = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx"

if not os.path.exists(MODEL_PATH):
    print("Downloading MNIST model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model ready.")

session = ort.InferenceSession(MODEL_PATH)
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("Session loaded.")
# ─────────────────────────────────────────────────────────────────────────────


def preprocess(image_data):
    """RGBA numpy → 1×1×28×28 float32 array for MNIST model."""
    pil = Image.fromarray(image_data.astype(np.uint8), "RGBA")
    bg  = Image.new("RGB", pil.size, (255, 255, 255))
    bg.paste(pil, mask=pil.split()[3])

    gray = ImageOps.grayscale(bg)
    if np.array(gray).mean() > 252:
        return None                          # blank canvas

    # Invert: MNIST expects white digit on black background
    gray = ImageOps.invert(gray)
    gray = gray.resize((28, 28), Image.LANCZOS)

    arr = np.array(gray, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 1, 28, 28)         # NCHW
    return arr


def predict(sketch):
    if sketch is None:
        return "✏️ Draw a digit first!", {}

    image_data = sketch.get("composite") if isinstance(sketch, dict) else sketch
    if image_data is None:
        return "✏️ Draw a digit first!", {}

    arr = preprocess(image_data)
    if arr is None:
        return "✏️ Canvas is empty — draw a digit!", {}

    logits = session.run([output_name], {input_name: arr})[0][0]  # shape (10,)

    # Softmax
    e = np.exp(logits - logits.max())
    probs = e / e.sum()

    top_idx  = int(probs.argmax())
    top_conf = float(probs[top_idx]) * 100
    conf_map = {str(i): round(float(probs[i]), 4) for i in range(10)}

    return f"## Predicted: `{top_idx}`\n**Confidence: {top_conf:.1f}%**", conf_map


# ── UI ────────────────────────────────────────────────────────────────────────
css = """
.title { text-align:center; font-size:2rem; font-weight:800; margin-bottom:0; }
.subtitle { text-align:center; color:#888; margin-top:4px; margin-bottom:24px; }
footer { display:none !important; }
"""

with gr.Blocks(css=css, title="Digit Recognizer") as demo:
    gr.HTML("<h1 class='title'>🔢 Digit Recognizer</h1>")
    gr.HTML("<p class='subtitle'>Draw any digit 0–9 and click <b>Predict</b></p>")

    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.Sketchpad(
                label="Draw here",
                type="numpy",
                brush=gr.Brush(
                    default_size=24,
                    colors=["#000000"],
                    default_color="#000000",
                ),
                canvas_size=(400, 400),
            )
            with gr.Row():
                btn_predict = gr.Button("Predict →", variant="primary", scale=2)
                btn_clear   = gr.ClearButton([canvas], value="Clear", scale=1)

        with gr.Column(scale=1):
            out_text  = gr.Markdown("*Draw a digit and click Predict*")
            out_label = gr.Label(label="Confidence", num_top_classes=10)

    btn_predict.click(fn=predict, inputs=[canvas], outputs=[out_text, out_label])

demo.launch()