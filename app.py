import os
import torch
from flask import Flask, request, render_template, jsonify
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
from googletrans import Translator

# Initialize Flask app
app = Flask(__name__, template_folder='template')

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize translator
translator = Translator()

# Load image captioning model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Simple responses for the chatbot
chat_responses = {
  "hello": "Hello! How can I help you today?",
  "hi": "Hi there! Want to analyze an image?",
  "help": "Upload an image and I'll describe what I see. You can also choose a language for the caption.",
  "who are you": "I'm an AI-powered image captioning assistant. I can analyze your images and tell you what I see!",
  "how does this work": "Upload an image, select your preferred language, and I'll generate a description of what's in the picture.",
  "languages": "You can get captions in English, Spanish, French, German, Chinese, Japanese, and more!",
  "thank you": "You're welcome! Feel free to upload more images anytime.",
  "thanks": "You're welcome! Is there anything else you'd like to know about the image?",
  "bye": "Goodbye! Come back when you have more images to analyze.",
  "can you describe this image": "Sure! Just upload an image, and I'll describe what's in it.",
  "what do you see in this picture": "Please upload the picture, and I'll tell you what I see!",
  "can you detect objects in photos": "Yes! I can recognize objects, scenes, and even people in photos.",
  "can you caption multiple images": "Currently, I process one image at a time. Upload one and I'll describe it for you!",
  "do you support other languages": "Yes! I can generate captions in English, Spanish, French, German, Japanese, Chinese, and more.",
  "can you translate the caption": "Absolutely! Just select your preferred language before or after uploading the image.",
  "how do I change the language": "You can choose a language option before submitting the image. I’ll then caption it in that language!",
  "what should I do first": "Just upload an image, and I’ll take care of the rest!",
  "how do I use this tool": "Upload your image, choose a language, and I’ll describe what’s in the photo for you.",
  "what is this tool for": "This tool helps generate descriptive captions for your images using AI.",
  "are you human": "Nope! I’m a helpful AI here to describe your pictures.",
  "do you have a name": "I'm your image captioning assistant — but you can give me a name if you'd like!",
  "can you tell jokes": "Sure! Why did the picture go to jail? Because it was framed!",
  "this is cool": "I'm glad you like it! Upload another image to try more.",
  "it’s not working": "Sorry to hear that! Try reloading the page or re-uploading the image.",
  "can I try another image": "Absolutely! Upload as many as you'd like — I'm ready!",
  "see you later": "See you! Don’t forget to come back with more images.",
  "good night": "Good night! Talk to you soon!",
  "i’m done": "Great! Let me know if you need anything else later."
}

def generate_caption(image_path, lang="en"):
    """Generate captions for the given image and translate if needed"""
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
    
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    if lang != "en":
        try:
            caption = translator.translate(caption, dest=lang).text
        except Exception as e:
            print(f"Translation error: {e}")
    
    return caption

def get_chatbot_response(user_query, image_caption=None):
    """Generate chatbot response based on user query and image caption"""
    user_query = user_query.lower().strip()
    
    # Check for predefined responses
    if user_query in chat_responses:
        return chat_responses[user_query]
    
    # If we have an image caption, try to answer questions about the image
    if image_caption:
        if "what" in user_query and ("in the image" in user_query or "in the picture" in user_query or "do you see" in user_query):
            return f"I can see {image_caption} in the image."
        elif "describe" in user_query or "tell me about" in user_query:
            return f"The image shows {image_caption}."
    
    # Default response
    return "I'm not sure how to answer that. Try uploading an image or asking me to describe what I see."

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Process image upload and generate caption"""
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"})
    
    lang = request.form.get("language", "en")
    
    # Save the uploaded file
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    
    # Generate caption
    caption = generate_caption(filename, lang)
    
    # Create response
    response = f"I see {caption}"
    
    return jsonify({
        "response": response,
        "caption": caption,
        "image_path": filename
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages from user"""
    data = request.json
    user_message = data.get('message', '')
    last_caption = data.get('lastCaption', None)
    
    response = get_chatbot_response(user_message, last_caption)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)