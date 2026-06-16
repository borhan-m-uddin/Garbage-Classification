import os
import tempfile

import gradio as gr

from model import GarbageClassifier

classifier = GarbageClassifier(
    "Garbage_Classification_Outputs/best_model.pth",
    classes_path="Garbage_Classification_Outputs/classes.json",
    device="cpu",
)


def predict_image(image):
    """Gradio interface for single image prediction."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        result = classifier.predict(tmp.name)

    os.unlink(tmp.name)

    output_lines = [
        f"Class: {result['class']}",
        f"Confidence: {result['confidence'] * 100:.2f}%",
        "",
        "All probabilities:",
    ]

    for class_name, prob in sorted(
        result["all_probabilities"].items(), key=lambda x: x[1], reverse=True
    ):
        output_lines.append(f"- {class_name}: {prob * 100:.2f}%")

    return "\n".join(output_lines)


interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="markdown",
    title="Garbage Classification AI",
    description="Upload an image of garbage and get a classification result.",
    examples=[],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    interface.launch()