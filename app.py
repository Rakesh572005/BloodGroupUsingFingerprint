import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from pyfingerprint.pyfingerprint import PyFingerprint
from PIL import Image, ImageEnhance
import numpy as np
import cv2  # For edge detection
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Initialize Flask app
app = Flask(__name__)

# Blood group prediction labels
disease_dic = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Load the machine learning model
model_path = 'model_blood.h5'  # Update with your model path
loaded_model_imageNet = load_model(model_path)

# Path to static directory for serving images
STATIC_FOLDER = 'static'
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

# Initialize the ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def check_sensor_connection():
    """Checks if the fingerprint sensor is connected and operational."""
    try:
        f = PyFingerprint('/dev/ttyUSB0', 57600, 0xFFFFFFFF, 0x00000000)
        if f.verifyPassword():
            return True
        else:
            return False
    except Exception:
        return False

def capture_fingerprint():
    """Captures fingerprint and saves it as an RGB image in the static folder."""
    try:
        # Initialize sensor
        f = PyFingerprint('/dev/ttyUSB0', 57600, 0xFFFFFFFF, 0x00000000)

        # Check if sensor is connected and password is correct
        if not f.verifyPassword():
            raise ValueError('The fingerprint sensor password is incorrect!')

        print('Fingerprint sensor connected and ready!')

        # Wait for finger placement
        print('Waiting for finger...')
        while not f.readImage():
            pass

        # Capture and save the fingerprint image
        temp_image_path = os.path.join(STATIC_FOLDER, 'fingerprint.bmp')
        f.downloadImage(temp_image_path)

        # Convert to RGB and save as PNG
        img = Image.open(temp_image_path).convert('RGB')
        rgb_image_path = os.path.join(STATIC_FOLDER, 'fingerprint_rgb.png')
        img.save(rgb_image_path)

        print(f"Fingerprint saved at {rgb_image_path}")
        return rgb_image_path  # Return the local path to the RGB image
    except Exception as e:
        raise RuntimeError(f"Fingerprint capture failed: {str(e)}")

def analyze_fingerprint(image_path):
    """Analyzes the fingerprint image for loops, arches, and whorls."""
    try:
        # Load image
        img = Image.open(image_path).convert('L')  # Convert to grayscale

        # Preprocessing (e.g., binarization, edge detection)
        img_array = np.array(img)
        _, binary_img = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(binary_img, 100, 200)

        # Placeholder feature extraction (replace with actual analysis)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        loops, arches, whorls = 0, 0, 0

        # Loop through contours to classify them based on shape features (rough estimation)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Arbitrary size filter for fingerprint features
                if area > 1000:
                    whorls += 1
                elif area > 500:
                    loops += 1
                else:
                    arches += 1

        return loops, arches, whorls
    except Exception as e:
        raise RuntimeError(f"Fingerprint analysis failed: {str(e)}")

def preprocess_image(image_path):
    """Preprocesses the image by performing data augmentation and edge detection."""
    try:
        # Load the image
        img = Image.open(image_path).convert('RGB')

        # Apply edge detection (Canny edge detection using OpenCV)
        img_array = np.array(img)
        edges = cv2.Canny(img_array, 100, 200)  # Perform edge detection

        # Convert back to PIL Image
        edges_image = Image.fromarray(edges)

        # Enhance the image (optional: adjust brightness, contrast, etc.)
        enhancer = ImageEnhance.Contrast(edges_image)
        enhanced_image = enhancer.enhance(2)  # Increase contrast

        # Convert enhanced image to numpy array
        augmented_image = np.array(enhanced_image)

        # Reshape for ImageDataGenerator (add batch dimension and channels)
        augmented_image = augmented_image.reshape((1,) + augmented_image.shape + (1,))  # Shape: (1, height, width, channels)

        # Apply data augmentation
        augmented_image_gen = datagen.flow(augmented_image, batch_size=1)

        # Get augmented image (use __next__ or next())
        augmented_image = next(augmented_image_gen)

        # Resize image to model input size (e.g., 256x256)
        augmented_image_resized = cv2.resize(augmented_image[0], (256, 256))

        # Add the missing channels dimension for RGB (convert to (256, 256, 3))
        augmented_image_resized_rgb = np.stack([augmented_image_resized] * 3, axis=-1)

        return augmented_image_resized_rgb
    except Exception as e:
        raise RuntimeError(f"Image preprocessing failed: {str(e)}")

def predict_blood_group(image_path):
    """Predicts the blood group from a processed fingerprint image."""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)

        # Convert the image to the format expected by the model (batch_size, height, width, channels)
        x = np.expand_dims(processed_image, axis=0)
        x = preprocess_input(x)

        # Predict using the loaded model
        result = loaded_model_imageNet.predict(x)
        final_list_result = (result * 100).astype('int')
        list_vals = list(final_list_result[0])
        result_val = max(list_vals)
        index_result = list_vals.index(result_val)

        return index_result
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html', title="Blood Group Detection")

def recommend_diet(loops, arches, whorls):
    """Recommends a diet plan based on fingerprint analysis."""
    if arches > loops and arches > whorls:
        return 'High-protein diet with leafy vegetables.'
    elif whorls > loops and whorls > arches:
        return 'Balanced diet with moderate carbs and proteins.'
    else:
        return 'Calorie-rich diet for better energy.'

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        try:
            # First check if the fingerprint sensor is available
            sensor_connected = check_sensor_connection()

            if not sensor_connected:
                return jsonify({"status": "error", "message": "Fingerprint sensor is not connected!"})

            # Capture fingerprint and get image path
            image_filename = capture_fingerprint()

            # Predict blood group based on the captured fingerprint image
            prediction_idx = predict_blood_group(image_filename)
            prediction = str(disease_dic[prediction_idx])

            # Analyze fingerprint for loops, arches, and whorls
            loops, arches, whorls = analyze_fingerprint(image_filename)

            # Get diet recommendation based on fingerprint analysis
            diet_plan = recommend_diet(loops, arches, whorls)

            # Return the captured image, diet recommendation, and prediction to be displayed
            return jsonify({
                "status": "captured",
                "image_path": f"/static/{os.path.basename(image_filename)}",
                "prediction": prediction,
                "diet_plan": diet_plan
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})

    return render_template('index.html', title="Capture Fingerprint")

# To serve fingerprint images
@app.route('/static/<filename>')
def uploaded_file(filename):
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=3838)
