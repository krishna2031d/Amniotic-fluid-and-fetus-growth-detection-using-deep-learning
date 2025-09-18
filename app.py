import os
import sqlite3
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, request, flash, redirect, url_for, session, g, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask App
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configurations
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------- Database Setup ----------------- #
DATABASE = "users.db"

def get_db():
    if not hasattr(g, '_database'):
        g._database = sqlite3.connect(DATABASE)
        g._database.row_factory = sqlite3.Row
    return g._database

def create_users_table():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT UNIQUE NOT NULL,
                            password TEXT NOT NULL)''')
        db.commit()

@app.teardown_appcontext
def close_connection(exception):
    if hasattr(g, '_database'):
        g._database.close()

# ----------------- Model Setup ----------------- #
try:
    model = tf.keras.models.load_model('fetus_growth_model.keras')
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def analyze_amniotic_fluid(image_path):
    """
    Analyzes the X-ray image to estimate the amniotic fluid level.
    Uses adaptive thresholding and morphological operations for accuracy.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0, "Error: Invalid Image"

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Adaptive thresholding for better segmentation
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours and analyze fluid area
    contours, _ = cv2.findContours(clean_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    total_pixels = np.count_nonzero(mask)
    fluid_pixels = np.count_nonzero(clean_thresh)

    # Prevent division by zero
    af_ratio = (fluid_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    
    # **Limit AFI volume to a maximum of 80%**
    af_ratio = min(af_ratio, 80)
    
    # **Categorizing AFI**
    if af_ratio < 5:
        condition = "Oligohydramnios (Low AFI)"
    elif 5 <= af_ratio <= 25:
        condition = "Normal AFI"
    else:
        condition = "Polyhydramnios (High AFI)"
    
    return round(af_ratio, 2), condition

def get_prediction_class(prediction):
    predicted_class_idx = np.argmax(prediction, axis=-1)[0]
    class_labels = ["Fetal Abdomen", "Fetal Brain", "Fetal Femur", "Fetal Thorax", "Class 5", "Class 6"]  # Update class names
    return f"Issue detected in formation of {class_labels[predicted_class_idx]}" if predicted_class_idx < len(class_labels) else "Uncertain"

# ----------------- Authentication Routes ----------------- #
@app.route('/', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # ✅ Check if it's JSON (fetch call)
        if request.is_json:
            data = request.get_json()
            username = data.get("username")
            password = data.get("password")
        else:
            # ✅ Traditional form fallback
            username = request.form.get("username")
            password = request.form.get("password")

        if not username or not password:
            return jsonify({"success": False, "message": "❌ All fields are required!"}), 400

        hashed_password = generate_password_hash(password)
        db = get_db()
        cursor = db.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            db.commit()
            return jsonify({"success": True, "message": "✅ Account created successfully!"})
        except sqlite3.IntegrityError:
            return jsonify({"success": False, "message": "❌ Username already exists!"}), 409

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user and check_password_hash(user["password"], password):
            session["user"] = username
            return redirect(url_for("upload"))
        flash("❌ Incorrect username or password!", "danger")
    return render_template("login.html")

@app.route('/upload', methods=['GET', 'POST'])
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        file = request.files.get('file')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if model:
                processed_image = preprocess_image(filepath)
                prediction = model.predict(processed_image)
                prediction_class = get_prediction_class(prediction)
            else:
                prediction_class = "⚠️ Model not available"

            # **Get AFI Volume & Condition**
            afi_percentage, afi_condition = analyze_amniotic_fluid(filepath)
            
            return render_template("upload.html", results={
                "image_url": url_for('static', filename=f"uploads/{filename}"),
                "status": prediction_class,
                "fluid_analysis": f"Estimated Amniotic Fluid: {afi_percentage}% ({afi_condition})",
                "afi_condition": afi_condition  # ✅ Display the condition
            })
    
    return render_template("upload.html")


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("✅ Logged out successfully!", "success")
    return redirect(url_for('login'))

# ----------------- Run Flask App ----------------- #
if __name__ == "__main__":
    create_users_table()
    app.run(debug=True)
