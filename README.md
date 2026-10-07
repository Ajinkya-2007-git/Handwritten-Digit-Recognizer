# 🔢 Digit Recognizer

A handwritten digit recognizer built with Gradio. Draw a digit from 0 to 9 on the canvas, click **Predict**, and a pretrained neural network tells you what it sees, along with a confidence score for every digit.

**Live demo:** https://ajinkya007-my-handwrittendigit-recognizer.hf.space

## Features

- Draw digits directly in the browser with an adjustable sketchpad
- Instant predictions with a per-digit confidence breakdown
- MNIST-style preprocessing (crop, scale to 20×20, center by center of mass) for better accuracy on hand-drawn input
- Modern, responsive UI with light and dark mode support
- The model downloads automatically on first run

## How it works

1. The canvas drawing is flattened onto a white background and converted to grayscale.
2. It is inverted to a white digit on black, matching the MNIST format.
3. The digit is cropped, resized so its longest side is 20 px, then centered in a 28×28 image.
4. The image is passed to the pretrained MNIST ONNX model through ONNX Runtime.
5. The output logits go through a softmax to produce confidence scores.

## Model

- **Model:** `mnist-12.onnx` from the [ONNX Model Zoo](https://github.com/onnx/models)
- **Input:** `1×1×28×28` grayscale image
- **Output:** 10 class scores (digits 0 to 9)

## Tech stack

- [Gradio](https://gradio.app/) for the UI
- [ONNX Runtime](https://onnxruntime.ai/) for inference
- [Pillow](https://python-pillow.org/) and [NumPy](https://numpy.org/) for image processing

## Run locally

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
pip install -r requirements.txt
python app.py
```

Then open the local URL shown in the terminal (usually http://127.0.0.1:7860).

### requirements.txt

```
gradio
numpy
pillow
onnxruntime
```

## Tips for best results

- Draw one digit at a time
- Draw it large with a thick stroke
- Use **Clear** between predictions

## Project structure

```
├── app.py            # Gradio app, preprocessing, and inference
├── requirements.txt  # Python dependencies
└── README.md
```

## Acknowledgements

- MNIST dataset by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- Pretrained model from the ONNX Model Zoo 
