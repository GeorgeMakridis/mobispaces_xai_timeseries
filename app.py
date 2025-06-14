from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from lime import lime_image, lime_text
from skimage.segmentation import mark_boundaries
import os
import json
import cv2
from PIL import Image
import skimage.io
import skimage.transform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, make_pipeline
import eli5
from eli5.lime import TextExplainer

app = Flask(__name__)

# Global variables
model_image = None
model_text = None
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['normal', 'anomaly']

# SAX parameters
SAX_ALPHABET_SIZE = 7
SAX_WORD_SIZE = 5

def create_sax_cuts(alphabet_size):
    """Create SAX cuts for given alphabet size"""
    if alphabet_size == 7:
        return [-1.07, -0.57, -0.18, 0, 0.18, 0.57, 1.07]
    elif alphabet_size == 5:
        return [-0.84, -0.25, 0, 0.25, 0.84]
    else:
        # Default to 5 alphabet size
        return [-0.84, -0.25, 0, 0.25, 0.84]

def znorm(ts):
    """Z-normalize time series"""
    ts = np.array(ts)
    return (ts - np.mean(ts)) / np.std(ts)

def ts_to_sax_string(ts, cuts):
    """Convert time series to SAX string"""
    normalized_ts = znorm(ts)
    sax_string = ""
    
    for value in normalized_ts:
        symbol_index = len([cut for cut in cuts if value > cut])
        sax_string += chr(ord('a') + symbol_index)
    
    return sax_string

def get_image_data(data):
    """Convert time series to image representation"""
    fig = plt.figure(figsize=(8, 6))
    plt.plot(data)
    plt.axis('off')
    
    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    try:
        img = skimage.io.imread(buf).astype(float)
        img = skimage.transform.resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant')
        
        if img is None or len(img.shape) < 2 or len(img.shape) > 3:
            return None
            
        if len(img.shape) == 2:
            img = np.tile(img[:, :, None], (1, 1, 3))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        elif img.shape[2] > 4:
            return None
            
        img = img / 255.0
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
    finally:
        plt.close(fig)
        buf.close()
    
    return img

def load_models():
    """Load both image and text models for time series analysis"""
    global model_image, model_text
    
    # Image-based CNN model
    model_image = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./1),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    
    model_image.build((None, IMG_HEIGHT, IMG_WIDTH, 3))
    model_image.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Text-based model (SAX + SVM)
    vec = TfidfVectorizer(min_df=0.0, stop_words=None, ngram_range=(1, 2), lowercase=False)
    svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
    lsa = make_pipeline(vec, svd)
    clf = SVC(C=10, gamma=0.1, probability=True, random_state=42)
    model_text = make_pipeline(lsa, clf)
    
    # Train text model with dummy data
    dummy_sax_data = generate_dummy_sax_data()
    model_text.fit(dummy_sax_data['texts'], dummy_sax_data['labels'])
    
    print("Models loaded successfully")

def generate_dummy_sax_data():
    """Generate dummy SAX data for model training"""
    texts = []
    labels = []
    
    # Generate dummy time series and convert to SAX
    for i in range(100):
        # Normal pattern
        ts = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
        sax_string = ts_to_sax_string(ts, create_sax_cuts(SAX_ALPHABET_SIZE))
        chunks = [sax_string[j:j+SAX_WORD_SIZE] for j in range(0, len(sax_string), SAX_WORD_SIZE)]
        texts.append(' '.join(chunks))
        labels.append(0)
        
        # Anomaly pattern
        ts = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.5, 100)
        ts[40:60] += 2  # Add anomaly
        sax_string = ts_to_sax_string(ts, create_sax_cuts(SAX_ALPHABET_SIZE))
        chunks = [sax_string[j:j+SAX_WORD_SIZE] for j in range(0, len(sax_string), SAX_WORD_SIZE)]
        texts.append(' '.join(chunks))
        labels.append(1)
    
    return {'texts': texts, 'labels': labels}

def preprocess_timeseries(ts_data):
    """Preprocess time series data"""
    if isinstance(ts_data, str):
        # Convert string to list of floats
        ts_data = [float(x) for x in ts_data.split(',')]
    
    ts_array = np.array(ts_data)
    
    # Generate image representation
    img_data = get_image_data(ts_array)
    
    # Generate SAX representation
    sax_string = ts_to_sax_string(ts_array, create_sax_cuts(SAX_ALPHABET_SIZE))
    chunks = [sax_string[i:i+SAX_WORD_SIZE] for i in range(0, len(sax_string), SAX_WORD_SIZE)]
    sax_text = ' '.join(chunks)
    
    return ts_array, img_data, sax_text

def make_gradcam_heatmap(img_array, model, pred_index=None):
    """Generate GradCAM heatmap for time series image"""
    # Find the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        raise ValueError("No convolutional layer found")
    
    # Create gradient model
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [last_conv_layer.output, model.output]
    )
    
    # Expand dimensions for batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Compute gradients
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def create_gradcam_visualization(img_array, heatmap):
    """Create GradCAM visualization"""
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))
    
    # Apply colormap
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    
    # Normalize image
    img_normalized = img_array.astype(np.float32)
    
    # Superimpose
    superimposed = heatmap_colored * 0.4 + img_normalized * 0.6
    
    return superimposed

def generate_lime_explanation(img_array, model):
    """Generate LIME explanation for time series image"""
    explainer = lime_image.LimeImageExplainer()
    
    def predict_fn(images):
        return model.predict(images)
    
    explanation = explainer.explain_instance(
        img_array.astype(np.double),
        predict_fn,
        top_labels=len(CLASS_NAMES),
        hide_color=0,
        num_samples=100
    )
    
    # Get image and mask for top prediction
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        hide_rest=False,
        num_features=10
    )
    
    # Create visualization
    img_boundary = mark_boundaries(temp, mask)
    
    return img_boundary

def generate_text_explanation(sax_text, model):
    """Generate text-based explanation using LIME"""
    te = TextExplainer(random_state=42)
    te.fit(sax_text, model.predict_proba)
    
    # Get explanation
    explanation = te.explain_instance(sax_text)
    
    return explanation

def array_to_base64(img_array):
    """Convert numpy array to base64 string"""
    # Ensure values are in 0-255 range
    img_array = (img_array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Save to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "image_model_loaded": model_image is not None,
        "text_model_loaded": model_text is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Basic prediction endpoint"""
    try:
        data = request.json
        ts_data = data['timeseries']  # comma-separated values or list
        
        # Preprocess time series
        ts_array, img_data, sax_text = preprocess_timeseries(ts_data)
        
        if img_data is None:
            return jsonify({"error": "Failed to process time series"}), 400
        
        # Make predictions
        img_batch = np.expand_dims(img_data, axis=0)
        img_predictions = model_image.predict(img_batch)
        text_predictions = model_text.predict_proba([sax_text])
        
        # Get results
        img_pred_class = np.argmax(img_predictions[0])
        img_confidence = float(img_predictions[0][img_pred_class])
        
        text_pred_class = np.argmax(text_predictions[0])
        text_confidence = float(text_predictions[0][text_pred_class])
        
        result = {
            "image_prediction": {
                "predicted_class": CLASS_NAMES[img_pred_class],
                "confidence": img_confidence,
                "all_predictions": {
                    CLASS_NAMES[i]: float(img_predictions[0][i]) 
                    for i in range(len(CLASS_NAMES))
                }
            },
            "text_prediction": {
                "predicted_class": CLASS_NAMES[text_pred_class],
                "confidence": text_confidence,
                "all_predictions": {
                    CLASS_NAMES[i]: float(text_predictions[0][i]) 
                    for i in range(len(CLASS_NAMES))
                }
            },
            "sax_representation": sax_text
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/gradcam', methods=['POST'])
def gradcam_explanation():
    """GradCAM explanation endpoint for time series"""
    try:
        data = request.json
        ts_data = data['timeseries']
        
        # Preprocess time series
        ts_array, img_data, sax_text = preprocess_timeseries(ts_data)
        
        if img_data is None:
            return jsonify({"error": "Failed to process time series"}), 400
        
        # Generate GradCAM
        heatmap = make_gradcam_heatmap(img_data, model_image)
        
        # Create visualization
        gradcam_viz = create_gradcam_visualization(img_data, heatmap)
        
        # Convert to base64
        gradcam_b64 = array_to_base64(gradcam_viz)
        
        # Get prediction info
        img_batch = np.expand_dims(img_data, axis=0)
        predictions = model_image.predict(img_batch)
        pred_class = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_class])
        
        result = {
            "gradcam_image": gradcam_b64,
            "original_plot": array_to_base64(img_data),
            "predicted_class": CLASS_NAMES[pred_class],
            "confidence": confidence,
            "explanation": "GradCAM highlights the regions in the time series plot that most influenced the model's prediction"
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/lime', methods=['POST'])
def lime_explanation():
    """LIME explanation endpoint for time series"""
    try:
        data = request.json
        ts_data = data['timeseries']
        
        # Preprocess time series
        ts_array, img_data, sax_text = preprocess_timeseries(ts_data)
        
        if img_data is None:
            return jsonify({"error": "Failed to process time series"}), 400
        
        # Generate LIME explanation
        lime_viz = generate_lime_explanation(img_data, model_image)
        
        # Convert to base64
        lime_b64 = array_to_base64(lime_viz)
        
        # Get prediction info
        img_batch = np.expand_dims(img_data, axis=0)
        predictions = model_image.predict(img_batch)
        pred_class = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_class])
        
        result = {
            "lime_image": lime_b64,
            "original_plot": array_to_base64(img_data),
            "predicted_class": CLASS_NAMES[pred_class],
            "confidence": confidence,
            "explanation": "LIME shows which parts of the time series plot contribute positively or negatively to the prediction"
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/text_explanation', methods=['POST'])
def text_explanation():
    """Text-based explanation using SAX representation"""
    try:
        data = request.json
        ts_data = data['timeseries']
        
        # Preprocess time series
        ts_array, img_data, sax_text = preprocess_timeseries(ts_data)
        
        # Generate text explanation
        te = TextExplainer(random_state=42)
        te.fit(sax_text, model_text.predict_proba)
        
        # Get prediction
        text_prediction = model_text.predict_proba([sax_text])
        pred_class = np.argmax(text_prediction[0])
        confidence = float(text_prediction[0][pred_class])
        
        # Get feature importances (simplified)
        feature_weights = te.feature_importances_
        
        result = {
            "predicted_class": CLASS_NAMES[pred_class],
            "confidence": confidence,
            "sax_representation": sax_text,
            "explanation": "Text explanation based on SAX (Symbolic Aggregate approXimation) representation",
            "feature_importance": "Feature weights show which SAX words contribute most to the prediction"
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    docs = {
        "API Documentation": {
            "description": "Time Series XAI Service - Explainable AI for Time Series Analysis",
            "endpoints": {
                "/health": "GET - Health check",
                "/predict": "POST - Time series classification (both image and text models)",
                "/gradcam": "POST - GradCAM explanation for image representation",
                "/lime": "POST - LIME explanation for image representation", 
                "/text_explanation": "POST - Text-based explanation using SAX representation"
            },
            "request_format": {
                "timeseries": "comma-separated values as string or array of numbers"
            },
            "example_request": {
                "timeseries": "1.2,2.3,1.8,2.1,1.9,2.5,1.7,2.0,1.8,2.2"
            },
            "features": [
                "Converts time series to image plots for CNN analysis",
                "Converts time series to SAX representation for text analysis",
                "Provides GradCAM heatmaps showing important time regions",
                "Provides LIME explanations for visual interpretability",
                "Provides text-based explanations using SAX symbolic representation"
            ]
        }
    }
    return jsonify(docs)

if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run app
    app.run(host='0.0.0.0', port=5000, debug=True)