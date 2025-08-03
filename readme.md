# Time Series XAI Docker Service

A comprehensive Dockerized Flask service that provides explainable AI (XAI) capabilities for time series analysis using multiple explanation techniques including GradCAM, LIME, and SAX-based text explanations.

## 🚀 Features

- **Dual Model Architecture**: Image-based CNN and text-based SAX+SVM models
- **Multiple Explanation Methods**:
  - **GradCAM**: Visual heatmaps showing important time regions in plot representations
  - **LIME**: Segment-based explanations for time series plots
  - **SAX Text Explanations**: Symbolic representation analysis using ELI5
- **RESTful API** with comprehensive endpoints
- **Dockerized deployment** with health checks
- **Time Series Preprocessing**: Automatic conversion to both image and symbolic representations

## 📊 How It Works

### 1. Time Series to Image Conversion
The service converts time series data into plot images using matplotlib, then applies CNN-based analysis with visual explainability techniques.

### 2. Time Series to SAX (Symbolic Aggregate approXimation)
Time series are converted to symbolic strings using SAX representation, enabling text-based machine learning and explanations.

### 3. Explanation Generation
- **GradCAM**: Highlights which parts of the time series plot influenced the prediction
- **LIME**: Shows positive/negative contributions of different plot regions
- **Text Analysis**: Explains which symbolic patterns in the SAX representation drove the decision

## 🛠️ API Endpoints

### Health Check
```http
GET /health
```
Returns service health status and model loading state.

**Response:**
```json
{
  "status": "healthy",
  "image_model_loaded": true,
  "text_model_loaded": true
}
```

### Time Series Prediction
```http
POST /predict
Content-Type: application/json

{
  "timeseries": "1.2,2.3,1.8,2.1,1.9,2.5,1.7,2.0,1.8,2.2"
}
```

**Response:**
```json
{
  "image_prediction": {
    "predicted_class": "normal",
    "confidence": 0.85,
    "all_predictions": {"normal": 0.85, "anomaly": 0.15}
  },
  "text_prediction": {
    "predicted_class": "normal", 
    "confidence": 0.78,
    "all_predictions": {"normal": 0.78, "anomaly": 0.22}
  },
  "sax_representation": "bccba ccbaa bccba ccbaa"
}
```

### GradCAM Explanation
```http
POST /gradcam
Content-Type: application/json

{
  "timeseries": "1.2,2.3,1.8,2.1,1.9,2.5,1.7,2.0,1.8,2.2"
}
```

**Response:**
```json
{
  "gradcam_image": "base64_encoded_heatmap_image",
  "original_plot": "base64_encoded_original_plot",
  "predicted_class": "normal",
  "confidence": 0.85,
  "explanation": "GradCAM highlights the regions in the time series plot that most influenced the model's prediction"
}
```

### LIME Explanation
```http
POST /lime
Content-Type: application/json

{
  "timeseries": "1.2,2.3,1.8,2.1,1.9,2.5,1.7,2.0,1.8,2.2"
}
```

**Response:**
```json
{
  "lime_image": "base64_encoded_explanation_image",
  "original_plot": "base64_encoded_original_plot", 
  "predicted_class": "normal",
  "confidence": 0.85,
  "explanation": "LIME shows which parts of the time series plot contribute positively or negatively to the prediction"
}
```

### Text-based SAX Explanation
```http
POST /text_explanation
Content-Type: application/json

{
  "timeseries": "1.2,2.3,1.8,2.1,1.9,2.5,1.7,2.0,1.8,2.2"
}
```

**Response:**
```json
{
  "predicted_class": "normal",
  "confidence": 0.78,
  "sax_representation": "bccba ccbaa bccba ccbaa",
  "explanation": "Text explanation based on SAX (Symbolic Aggregate approXimation) representation",
  "feature_importance": "Feature weights show which SAX words contribute most to the prediction"
}
```

## 📋 Quick Start

### 1. Build and Run with Docker Compose

```bash
# Create project directory
mkdir timeseries-xai && cd timeseries-xai

# Create all necessary files (see file structure below)

# Build and run the service
docker-compose -f docker-compose-timeseries.yml up --build
```

### 2. Alternative: Build and Run with Docker

```bash
# Build the image
docker build -f Dockerfile -t timeseries-xai .

# Run the container
docker run -p 5000:5000 timeseries-xai
```

### 3. Test the Service

```bash
# Install test dependencies
pip install requests pillow numpy

# Run the test client
python test_timeseries_client.py
```

## 📁 Project Structure

```
timeseries-xai/
├── timeseries_app.py              # Main Flask application
├── Dockerfile                     # Docker configuration
├── docker-compose-timeseries.yml  # Docker Compose setup
├── requirements_timeseries.txt    # Python dependencies
├── test_timeseries_client.py      # Test client script
├── README_TimeSeries_XAI.md      # This documentation
├── models/                       # Directory for model weights (optional)
└── sample_data/                  # Directory for sample time series data
```

## 🔧 Input Format

The service accepts time series data in two formats:

### 1. Comma-separated String
```json
{
  "timeseries": "1.2,2.3,1.8,2.1,1.9,2.5,1.7,2.0,1.8,2.2"
}
```

### 2. Array of Numbers
```json
{
  "timeseries": [1.2, 2.3, 1.8, 2.1, 1.9, 2.5, 1.7, 2.0, 1.8, 2.2]
}
```

## 🧪 Testing Examples

### Example 1: Normal Time Series
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"timeseries": "0.1,0.2,0.15,0.25,0.2,0.3,0.25,0.35,0.3,0.4"}'
```

### Example 2: Anomalous Time Series
```bash
curl -X POST http://localhost:5000/gradcam \
  -H "Content-Type: application/json" \
  -d '{"timeseries": "0.1,0.2,0.15,5.0,0.2,0.3,0.25,0.35,0.3,0.4"}'
```

### Example 3: Trending Time Series
```bash
curl -X POST http://localhost:5000/lime \
  -H "Content-Type: application/json" \
  -d '{"timeseries": "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"}'
```

## 🔍 Understanding the Explanations

### GradCAM Visualizations
- **Red/Yellow regions**: High importance areas that strongly influenced the prediction
- **Blue/Purple regions**: Low importance areas with minimal impact
- **Overlay intensity**: Indicates the strength of the influence

### LIME Explanations
- **Green segments**: Contribute positively to the predicted class
- **Red segments**: Contribute negatively to the predicted class
- **Boundary clarity**: Shows how the model segments the time series

### SAX Text Explanations
- **SAX symbols**: Letters (a-g) represent different value ranges
- **Word importance**: Shows which symbolic patterns are most predictive
- **Feature weights**: Quantify the contribution of each SAX word

## ⚙️ Configuration

### Model Customization
To use your own trained models, modify the `load_models()` function in `timeseries_app.py`:

```python
def load_models():
    global model_image, model_text
    
    # Load your custom image model
    model_image = tf.keras.models.load_model('path/to/your/image_model.h5')
    
    # Load your custom text model
    with open('path/to/your/text_model.pkl', 'rb') as f:
        model_text = pickle.load(f)
```

### SAX Parameters
Adjust SAX representation parameters:

```python
SAX_ALPHABET_SIZE = 7  # Number of symbols (a-g)
SAX_WORD_SIZE = 5      # Length of each SAX word
```

### Class Names
Update class names to match your problem:

```python
CLASS_NAMES = ['normal', 'anomaly', 'outlier', 'trend']
```

### Image Dimensions
Modify plot image dimensions:

```python
IMG_HEIGHT = 224
IMG_WIDTH = 224
```

## 🚀 Advanced Usage

### Batch Processing
```python
import requests

# Process multiple time series
time_series_batch = [
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [1, 1, 1, 1, 1]
]

results = []
for ts in time_series_batch:
    response = requests.post(
        'http://localhost:5000/predict',
        json={'timeseries': ts}
    )
    results.append(response.json())
```

### Custom Preprocessing
```python
import numpy as np

def preprocess_your_data(raw_data):
    # Apply your custom preprocessing
    normalized = (raw_data - np.mean(raw_data)) / np.std(raw_data)
    return normalized.tolist()

# Use with API
preprocessed_ts = preprocess_your_data(your_raw_data)
response = requests.post(
    'http://localhost:5000/gradcam',
    json={'timeseries': preprocessed_ts}
)
```

## 📊 Sample Data Generation

The service includes built-in sample data generation:

```python
def generate_sample_data():
    # Normal sine wave
    normal = np.sin(np.linspace(0, 4*np.pi, 100))
    
    # Anomalous with spike
    anomaly = np.sin(np.linspace(0, 4*np.pi, 100))
    anomaly[40:50] += 3
    
    # Trending data
    trend = np.linspace(0, 2, 100) + np.sin(np.linspace(0, 8*np.pi, 100)) * 0.5
    
    return normal, anomaly, trend
```

## 🔧 Troubleshooting

### Common Issues

1. **Service not starting**
   ```bash
   # Check Docker logs
   docker-compose -f docker-compose-timeseries.yml logs -f
   ```

2. **Memory issues with large time series**
   ```python
   # Downsample your data
   ts_downsampled = ts[::10]  # Take every 10th point
   ```

3. **Poor image quality**
   ```python
   # Increase plot resolution in get_image_data()
   fig = plt.figure(figsize=(12, 8), dpi=150)
   ```

4. **SAX conversion errors**
   ```python
   # Ensure time series has sufficient length
   min_length = 20
   if len(ts) < min_length:
       ts = np.pad(ts, (0, min_length - len(ts)), 'edge')
   ```

## 🔒 Production Considerations

1. **Security**: Add authentication and rate limiting
2. **Scaling**: Use multiple replicas with load balancer
3. **Monitoring**: Implement comprehensive logging and metrics
4. **Data Validation**: Add input validation and sanitization
5. **Model Management**: Implement model versioning and A/B testing
6. **Caching**: Cache SAX representations and image conversions
7. **Resource Limits**: Set appropriate memory and CPU limits

## 📈 Performance Optimization

```python
# Enable model caching
@lru_cache(maxsize=128)
def cached_sax_conversion(ts_tuple):
    return ts_to_sax_string(list(ts_tuple), create_sax_cuts(SAX_ALPHABET_SIZE))

# Batch processing for multiple predictions
def batch_predict(time_series_list):
    # Process multiple time series efficiently
    pass
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## Disclaimer
This software has been developed thanks to the MobiSpaces project (Grant agreement ID: 101070279)

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review Docker logs for errors
- Ensure all dependencies are correctly installed
- Verify input data format matches expected schema

## 🎯 Use Cases

This service is ideal for:
- **Anomaly Detection**: Identifying unusual patterns in time series
- **Predictive Maintenance**: Understanding failure patterns in sensor data
- **Financial Analysis**: Explaining market trend predictions
- **IoT Monitoring**: Interpreting sensor data classifications
- **Healthcare**: Analyzing patient vital signs and medical time series
- **Energy Management**: Understanding consumption pattern classifications
