# ♻️ Garbage Classification AI

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://borhan72-garbage-classification.hf.space/)
[![GitHub repo](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/borhan-m-uddin/Garbage-Classification)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

This project uses deep learning to classify images of waste into **7 categories**:

- Battery  
- Cardboard  
- Clothes  
- Glass  
- Metal  
- Paper  
- Plastic  

The model is built with **PyTorch** and **ResNet‑50** (transfer learning) and is deployed as both:

- 🖥️ A **Flask web application** (with HTML templates)
- 🤗 A **Gradio interface** on Hugging Face Spaces

---

## 🚀 Live Demo

Try it now without installing anything:  
👉 **[Garbage Classification AI on Hugging Face Spaces](https://borhan72-garbage-classification.hf.space/)**

---

## ✨ Features

- **Image Upload** – Drag & drop or select an image of waste.
- **Real‑time Prediction** – Get the predicted class and confidence score.
- **Top‑3 Probabilities** – View the most likely categories with percentage scores.
- **Two Interfaces** – Choose between a clean web UI (Flask) or a Gradio interface.
- **Lightweight & Fast** – Optimized for CPU inference.

---

## 🛠️ Technologies Used

| Layer       | Technology |
|-------------|------------|
| **Backend** | Python 3.10, PyTorch, Flask, Gunicorn |
| **Frontend**| HTML, CSS (Flask), Gradio |
| **Model**   | ResNet‑50 (pre‑trained on ImageNet, fine‑tuned) |
| **Deployment** | Hugging Face Spaces (Gradio SDK) |
| **Version Control** | Git & GitHub |

---





---

## 🧠 Model Details

- **Architecture** – ResNet‑50 (pre‑trained, fine‑tuned on the dataset).
- **Dataset** – ~5,000 images across 7 categories.
- **Performance** – Achieved **~92% validation accuracy** (confusion matrix and training metrics available locally).
- **Inference Device** – CPU (optimised for low‑resource environments).

---

## 💻 Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/borhan-m-uddin/Garbage-Classification.git
cd Garbage-Classification

## 📁 Project Structure
