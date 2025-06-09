
# 🖼️ AI Image Caption Generator (Flask App)

This project is a Flask-based web application that generates descriptive captions for uploaded images using a pre-trained Vision Transformer + GPT2 model (`nlpconnect/vit-gpt2-image-captioning`). It also supports caption translation into different languages using Google Translate.

---

## 📌 Features

* 🌐 Web-based UI for uploading images.
* 🧠 Uses a ViT-GPT2 model to generate natural language image captions.
* 🌍 Automatically translates the caption into multiple languages using Google Translate.
* 🤖 Built-in chatbot for basic interaction.

---

## 🚀 Demo Flow

1. Upload an image through the web interface.
2. The model generates a descriptive caption in English.
3. Optionally, the caption is translated into a user-selected language.
4. A chatbot answers basic questions like “hello,” “help,” “who are you,” etc.

---

## 🛠️ Technologies Used

| Technology                  | Purpose              |
| --------------------------- | -------------------- |
| Python                      | Backend logic        |
| Flask                       | Web server           |
| Transformers (Hugging Face) | Caption generation   |
| ViT-GPT2                    | Vision-to-Text model |
| Googletrans                 | Translation API      |
| Torch                       | Model computation    |
| PIL                         | Image processing     |

---

## 📁 Project Structure

```
flask_trail_caption generator/
│
├── app.py                        # Flask application logic
├── model_13.h5 / mymodel.h5      # (Not used in app.py currently; maybe for experimentation)
│
├── template/
│   └── index.html               # Frontend template
│
├── static/
│   ├── *.jpg / *.png           # Example images
│   └── uploads/                # Uploaded images
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/flask-image-caption-generator.git
cd flask-image-caption-generator
```

### 2. Install Dependencies

Use pip to install required packages:

```bash
pip install flask torch transformers pillow googletrans==4.0.0-rc1
```

### 3. Run the App

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## 💬 Example Prompts for Chatbot

* `hello` → Hello! How can I help you today?
* `who are you` → I’m an AI-powered image captioning assistant.
* `languages` → You can get captions in Hindi, Marathi, French, etc.

---

---

## 📄 License

This project is for educational and research purposes only.


