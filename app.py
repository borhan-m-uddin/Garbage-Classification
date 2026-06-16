import base64
import os
import uuid

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from model import GarbageClassifier

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = "uploads"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

classifier = GarbageClassifier(
    "Garbage_Classification_Outputs/best_model.pth",
    classes_path="Garbage_Classification_Outputs/classes.json",
    device="cpu",
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, GIF, or BMP."}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

    try:
        file.save(filepath)
        result = classifier.predict(filepath)

        with open(filepath, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        return jsonify(
            {
                "success": True,
                "class": result["class"],
                "confidence": f"{result['confidence'] * 100:.2f}%",
                "probabilities": result["all_probabilities"],
                "image": f"data:image/jpeg;base64,{img_base64}",
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
