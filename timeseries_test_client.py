import requests
import json
import numpy as np
import base64
from PIL import Image
import io

def generate_sample_timeseries():
    """Generate sample time series data"""
    # Normal sine wave
    normal_ts = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
    
    # Anomalous time series with spike
    anomaly_ts = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
    anomaly_ts[40:50] += 3  # Add anomaly spike
    
    # Trend time series
    trend_ts = np.linspace(0, 2, 100) + np.sin(np.linspace(0, 8*np.pi, 100)) * 0.5
    
    return {
        'normal': normal_ts.tolist(),
        'anomaly': anomaly_ts.tolist(), 
        'trend': trend_ts.tolist()
    }

def save_base64_image(b64_string, filename):
    """Save base64 encoded image to file"""
    if b64_string:
        image_data = base64.b64decode(b64_string)
        with open(filename, 'wb') as f:
            f.write(image_data)
        print(f"📸 Saved: {filename}")

def test_api():
    """Test the time series XAI API endpoints"""
    base_url = "http://localhost:5000"
    
    # Generate sample data
    sample_data = generate_sample_timeseries()
    
    print("🚀 Testing Time Series XAI Service")
    print("=" * 50)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test each time series type
    for ts_type, ts_data in sample_data.items():
        print(f"\n🔍 Testing with {ts_type} time series...")
        
        # Convert to comma-separated string
        ts_string = ','.join(map(str, ts_data))
        data = {"timeseries": ts_string}
        
        # Test basic prediction
        print(f"  Testing prediction for {ts_type}...")
        try:
            response = requests.post(f"{base_url}/predict", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Image Prediction: {result['image_prediction']['predicted_class']} "
                      f"(confidence: {result['image_prediction']['confidence']:.3f})")
                print(f"  ✅ Text Prediction: {result['text_prediction']['predicted_class']} "
                      f"(confidence: {result['text_prediction']['confidence']:.3f})")
                print(f"  📝 SAX representation: {result['sax_representation'][:50]}...")
            else:
                print(f"  ❌ Prediction failed: {response.text}")
        except Exception as e:
            print(f"  ❌ Prediction error: {e}")
        
        # Test GradCAM
        print(f"  Testing GradCAM for {ts_type}...")
        try:
            response = requests.post(f"{base_url}/gradcam", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ GradCAM - Class: {result['predicted_class']}, "
                      f"Confidence: {result['confidence']:.3f}")
                
                # Save images
                save_base64_image(result['gradcam_image'], f'gradcam_{ts_type}.png')
                save_base64_image(result['original_plot'], f'original_{ts_type}.png')
                
            else:
                print(f"  ❌ GradCAM failed: {response.text}")
        except Exception as e:
            print(f"  ❌ GradCAM error: {e}")
        
        # Test LIME
        print(f"  Testing LIME for {ts_type}...")
        try:
            response = requests.post(f"{base_url}/lime", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ LIME - Class: {result['predicted_class']}, "
                      f"Confidence: {result['confidence']:.3f}")
                
                # Save image
                save_base64_image(result['lime_image'], f'lime_{ts_type}.png')
                
            else:
                print(f"  ❌ LIME failed: {response.text}")
        except Exception as e:
            print(f"  ❌ LIME error: {e}")
        
        # Test text explanation
        print(f"  Testing text explanation for {ts_type}...")
        try:
            response = requests.post(f"{base_url}/text_explanation", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Text Explanation - Class: {result['predicted_class']}, "
                      f"Confidence: {result['confidence']:.3f}")
                print(f"  📝 SAX: {result['sax_representation'][:30]}...")
                
            else:
                print(f"  ❌ Text explanation failed: {response.text}")
        except Exception as e:
            print(f"  ❌ Text explanation error: {e}")

def test_with_custom_data():
    """Test with custom time series data"""
    print("\n🔧 Testing with custom data...")
    
    # Create custom time series
    custom_ts = [1.0, 1.2, 1.1, 1.3, 1.2, 1.5, 1.4, 1.6, 1.8, 2.0,
                 2.2, 2.1, 1.9, 1.7, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8]
    
    data = {"timeseries": custom_ts}
    
    try:
        response = requests.post("http://localhost:5000/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Custom data prediction successful")
            print(f"Image model: {result['image_prediction']['predicted_class']}")
            print(f"Text model: {result['text_prediction']['predicted_class']}")
        else:
            print(f"❌ Custom data failed: {response.text}")
    except Exception as e:
        print(f"❌ Custom data error: {e}")

if __name__ == "__main__":