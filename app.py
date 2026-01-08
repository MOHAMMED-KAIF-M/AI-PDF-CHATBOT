from flask import Flask, request, render_template, jsonify, session, redirect, url_for, flash
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image
from functools import wraps
# For the login_required decorator

# GPU Configuration
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU enabled for inference")
    except RuntimeError as e:
        print(f"⚠️ {e}")

app = Flask(__name__, template_folder='templates')

# --- NEW: SESSION CONFIG ---
app.secret_key = 'your_secret_key_here'  # Change this to a random string in production!
# Optional: Use Flask-Session for better persistence (pip install flask-session)
# from flask_session import Session
# app.config['SESSION_TYPE'] = 'filesystem'
# Session(app)

# --- CONFIG ---
MODEL_PATH = r"model_best_final_1.h5"
IMAGE_SIZE = (224, 224)

# --- LOAD MODEL ---
print("🔹 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# --- CLASS LABELS ---
class_map = {
    0: 'glioma_tumor',
    1: 'meningioma_tumor',
    2: 'no_tumor',
    3: 'pituitary_tumor'
}

# --- NEW: LOGIN REQUIRED DECORATOR ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- NEW: LOGIN ROUTES ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Updated hardcoded check (use 'user123' and 'user@123')
        if username == 'kaifu123' and password == 'kaifu@123':
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# --- EXISTING ROUTES (PROTECTED) ---
@app.route('/')
@login_required  # NEW: Require login to access home
def index():
    return render_template('index.html')

# --- IMAGE PREDICTION ---
@app.route('/predict', methods=['POST'])
@login_required  # NEW: Require login for predictions
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Cannot decode image'}), 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
    img_resized = img_resized / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    preds = model.predict(img_input)
    idx = int(np.argmax(preds))
    label = class_map[idx]
    confidence = float(np.max(preds))

    # XAI placeholder: base64 image
    pil_img = Image.fromarray((img_resized * 255).astype(np.uint8))
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    xai_image_url = f"data:image/png;base64,{img_base64}"

    return jsonify({'label': label, 'confidence': confidence, 'xai_image_url': xai_image_url})

# --- CHATBOT ---
@app.route('/chat', methods=['POST'])
@login_required  # NEW: Require login for chatbot
def chat():
    data = request.json
    user_msg = data.get('message', '').lower()

    # --- Brain ---
    if "brain" in user_msg or "what is brain" in user_msg:
        reply = ["The brain is the central organ of the nervous system controlling thoughts, memory, emotions, movement, and sensory processing."]
    # --- Parts of Brain ---
    elif "parts of brain" in user_msg or "brain parts" in user_msg or "parts in brain" in user_msg:
        reply = ["cerebrum,cerebellum,brainstem"]
    # --- Tumor types ---
    elif "cerebrum" in user_msg:
        reply = ["part of brain"]
    elif "tumor types" in user_msg or "types of tumor" in user_msg:
        reply = [
            "1. Pituitary: A tumor in the pituitary gland affecting hormone levels.\n",
            "2. Meningioma: A tumor forming in the protective layers of the brain, usually benign.\n",
            "3. Glioma: A tumor originating in glial cells that support neurons.\n",
            "4. Schwannoma: Tumor in nerve sheath cells affecting nerves.\n",
            "5. Medulloblastoma: Fast-growing cancer in the cerebellum, mainly in children.\n"
        ]

    # --- Scans ---
    elif "scan" in user_msg or "imaging" in user_msg:
        reply = [
            "1. MRI: Magnetic Resonance Imaging – detailed brain imaging using magnets.\n",
            "2. CT: Computed Tomography – cross-sectional X-ray images of the brain.\n",
            "3. PET: Positron Emission Tomography – shows glucose metabolism in brain cells.\n",
            "4. Biopsy: Small tissue sample taken to check if tumor is cancerous.\n"
        ]

    # --- Treatments ---
    elif "treatment" in user_msg:
        reply = [
            "1. Surgery – remove tumor physically.\n",
            "2. Radiation – high-energy beams to kill tumor cells.\n",
            "3. Chemotherapy – drugs to destroy cancer cells.\n",
            "4. Targeted therapy – drugs targeting tumor-specific proteins.\n",
            "5. Immunotherapy – boosts immune system to attack tumor cells.\n"
        ]

    # --- Therapy ---
    elif "therapy" in user_msg:
        reply = ["Therapy helps patients recover physical, cognitive, or emotional functions after treatment.\n"]

    # --- Diet for tumor types ---
    elif "diet for glioma" in user_msg or "glioma diet" in user_msg:
        reply = [
            "1. Leafy greens (spinach, kale) for vitamins and minerals.\n",
            "2. Berries for antioxidants.\n",
            "3. Omega-3 rich fish (salmon, mackerel).\n",
            "4. Nuts and seeds for healthy fats.\n",
            "5. Whole grains for sustained energy.\n"
        ]

    elif "diet for pituitary" in user_msg or "pituitary diet" in user_msg:
        reply = [
            "1. Leafy greens (spinach, kale) for vitamins and minerals.\n",
            "2. Berries for antioxidants.\n",
            "3. Omega-3 rich fish (salmon, mackerel).\n",
            "4. Nuts and seeds for healthy fats.\n",
            "5. Whole grains for sustained energy.\n"
        ]

    elif "diet for meningioma" in user_msg or "meningioma diet" in user_msg:
        reply = [
            "1. Cruciferous vegetables (broccoli, cauliflower, cabbage).\n",
            "2. Leafy greens (spinach, kale).\n",
            "3. Berries and citrus fruits for antioxidants.\n",
            "4. Green tea for polyphenols.\n",
            "5. Avoid processed foods and sugary drinks.\n"
        ]

    elif "diet for schwannoma" in user_msg or "schwannoma diet" in user_msg:
        reply = [
            "1. Foods rich in B vitamins (whole grains, eggs, dairy).\n",
            "2. Leafy greens and vegetables for nerve support.\n",
            "3. Fatty fish (salmon, sardines) for brain and nerve health.\n",
            "4. Nuts and seeds for antioxidants.\n",
            "5. Avoid excessive alcohol and processed foods.\n"
        ]

    elif "diet for medulloblastoma" in user_msg or "medulloblastoma diet" in user_msg:
        reply = [
            "1. Fruits and vegetables for overall immunity.\n",
            "2. Omega-3 rich fish and flaxseeds.\n",
            "3. Whole grains for energy.\n",
            "4. Dairy or fortified alternatives for calcium and vitamin D.\n",
            "5. Hydration and frequent small meals.\n"
        ]

    elif "diet" in user_msg:
        reply = [
            "1. Leafy greens (spinach, kale) for antioxidants.\n",
            "2. Berries for brain health.\n",
            "3. Fish (salmon, mackerel) for omega-3.\n",
            "4. Nuts for healthy fats and nutrients.\n",
            "5. Whole grains for energy and brain function.\n"
        ]

    # --- Prevention ---
    elif "prevention" in user_msg:
        reply = [
            "1. Limit radiation exposure.\n",
            "2. Avoid smoking and toxic chemicals.\n",
            "3. Maintain healthy diet and exercise.\n",
            "4. Regular health check-ups.\n",
            "5. Boost immunity and reduce stress.\n"
        ]

    elif "tumor" in user_msg:
        reply = ["A tumor is an abnormal growth of cells in the body. In the brain, it can be benign (non-cancerous) or malignant (cancerous) and may affect normal brain function depending on its size and location."]

    elif "hello" in user_msg or "hey" in user_msg or "hi" in user_msg:
        reply = ["Hello! 👋 Ask me anything about the brain, tumors, scans, treatments, diet, or prevention.\n"]

    elif "thank you" in user_msg:
        reply = ["Your welcome!, ask me if you have any other doubts, queries"]

    else:
        reply = ["I’m not sure I understood that 🤔. Try asking 'Tumor types', 'Scans', 'Treatments', 'Diet', or 'Prevention'.\n"]

    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True)