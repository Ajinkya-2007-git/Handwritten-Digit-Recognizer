import gradio as gr
import numpy as np
from PIL import Image, ImageOps
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

# If predictions are still bad, flip this to True (scales pixels to 0–1)
NORMALIZE_TO_01 = False
# ─────────────────────────────────────────────────────────────────────────────


def preprocess(image_data):
    """RGBA numpy → 1×1×28×28 float32 array, MNIST-style framing."""
    pil = Image.fromarray(image_data.astype(np.uint8), "RGBA")
    bg  = Image.new("RGB", pil.size, (255, 255, 255))
    bg.paste(pil, mask=pil.split()[3])

    gray = ImageOps.invert(ImageOps.grayscale(bg))   # white digit on black
    a = np.array(gray)

    ys, xs = np.where(a > 30)
    if len(ys) == 0:
        return None                                   # blank canvas

    # Crop to the digit, scale longest side to 20 px
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    s = 20 / max(h, w)
    small = Image.fromarray(a).resize(
        (max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS
    )

    # Paste into 28×28, then shift so center of mass is at the middle
    canvas = np.zeros((28, 28), np.float32)
    nw, nh = small.size
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top:top + nh, left:left + nw] = np.array(small, dtype=np.float32)

    total = canvas.sum()
    cy = (canvas.sum(axis=1) * np.arange(28)).sum() / total
    cx = (canvas.sum(axis=0) * np.arange(28)).sum() / total
    canvas = np.roll(canvas, (int(round(13.5 - cy)), int(round(13.5 - cx))), axis=(0, 1))

    if NORMALIZE_TO_01:
        canvas = canvas / 255.0
    return canvas.reshape(1, 1, 28, 28).astype(np.float32)


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
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

css = """
.gradio-container { max-width: 1000px !important; margin: auto !important; }
/* Hero header */
.hero {
    text-align: center;
    padding: 28px 16px 20px;
    margin-bottom: 18px;
    border-radius: 20px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 55%, #ec4899 100%);
    color: white;
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.35);
}
.hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; }
.hero p  { margin: 8px 0 0; opacity: 0.92; font-size: 1.02rem; }
/* Cards */
.card {
    border: 1px solid var(--border-color-primary);
    border-radius: 18px !important;
    padding: 16px !important;
    background: var(--background-fill-primary);
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.06);
}
.card-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 6px; }
/* Prediction result */
.result-box {
    text-align: center;
    border-radius: 16px;
    padding: 12px;
    background: linear-gradient(135deg, rgba(99,102,241,.12), rgba(236,72,153,.12));
    border: 1px solid rgba(99,102,241,.25);
}
.result-box h2 { font-size: 2.2rem; margin: 4px 0; }
/* Buttons */
#predict-btn { font-weight: 700; font-size: 1.05rem; box-shadow: 0 6px 16px rgba(99,102,241,.35); }
#predict-btn:hover { transform: translateY(-1px); transition: .15s; }
.tips { font-size: .9rem; opacity: .8; text-align: center; margin-top: 10px; }
footer { display: none !important; }
"""

with gr.Blocks(theme=theme, css=css, title="Digit Recognizer") as demo:
    gr.HTML(
        """
        <div class="hero">
            <h1>🔢 Digit Recognizer</h1>
            <p>Draw any digit from 0–9 and let the neural network guess it.</p>
        </div>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_classes="card"):
            gr.HTML("<div class='card-title'>🎨 Canvas</div>")
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
                btn_predict = gr.Button(
                    "Predict →", variant="primary", scale=2, elem_id="predict-btn"
                )
                btn_clear = gr.ClearButton([canvas], value="🗑️ Clear", scale=1)
            gr.HTML(
                "<div class='tips'>💡 Tip: draw one big, centered digit with a thick stroke for best results.</div>"
            )

        with gr.Column(scale=1, elem_classes="card"):
            gr.HTML("<div class='card-title'>🧠 Prediction</div>")
            out_text = gr.Markdown(
                "*Draw a digit and click Predict*", elem_classes="result-box"
            )
            out_label = gr.Label(label="Confidence per digit", num_top_classes=10)

    btn_predict.click(fn=predict, inputs=[canvas], outputs=[out_text, out_label])

demo.launch()
