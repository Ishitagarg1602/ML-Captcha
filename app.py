# Flask app.py with real-time prediction
from flask import Flask, request, jsonify, send_from_directory
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('captcha_model.pkl')
    with open('model_features.txt', 'r') as f:
        required_features = [line.strip() for line in f.readlines()]
    print("Model loaded successfully!")
except:
    print("Warning: Model not found. Running in data collection mode only.")
    model = None
    required_features = []

def extract_features(data):
    """Extract the same features used during training from real-time data"""
    features = {}
    
    # Mouse dynamics
    mouse_events = data.get('mouseEvents', [])
    if len(mouse_events) >= 3:
        df = pd.DataFrame(mouse_events)
        df['timestamp_diff'] = df['timestamp'].diff().fillna(0)
        df['distance'] = np.sqrt(df['x'].diff()**2 + df['y'].diff()**2).fillna(0)
        df['speed'] = np.where(df['timestamp_diff'] > 0, df['distance'] / df['timestamp_diff'], 0)
        df['acceleration'] = df['speed'].diff().fillna(0)
        
        # Calculate curvature
        curvatures = []
        for i in range(1, len(df) - 1):
            if i > 0 and i < len(df) - 1:
                dx1, dy1 = df.iloc[i]['x'] - df.iloc[i-1]['x'], df.iloc[i]['y'] - df.iloc[i-1]['y']
                dx2, dy2 = df.iloc[i+1]['x'] - df.iloc[i]['x'], df.iloc[i+1]['y'] - df.iloc[i]['y']
                
                if dx1 == 0 or dx2 == 0:
                    curvatures.append(0)
                    continue
                
                angle1 = np.arctan(dy1 / dx1)
                angle2 = np.arctan(dy2 / dx2)
                curvature = abs(angle2 - angle1)
                curvatures.append(curvature)
                
        features.update({
            'mouse_count': len(df),
            'mouse_mean_speed': df['speed'].mean(),
            'mouse_std_speed': df['speed'].std(),
            'mouse_max_speed': df['speed'].max(),
            'mouse_acceleration_var': df['acceleration'].var(),
            'mouse_path_length': df['distance'].sum(),
            'mouse_mean_curvature': np.mean(curvatures) if curvatures else 0,
            'mouse_std_curvature': np.std(curvatures) if curvatures else 0,
            'mouse_x_std': df['x'].std(),
            'mouse_y_std': df['y'].std(),
        })
        
    # Keyboard dynamics
    key_events = data.get('keyEvents', [])
    if len(key_events) >= 2:
        df = pd.DataFrame(key_events)
        
        keystroke_intervals = []
        backspace_count = 0
        error_indicators = 0
        
        for i in range(1, len(df)):
            if df.iloc[i-1]['eventType'] == 'keydown' and df.iloc[i]['eventType'] == 'keyup' and df.iloc[i-1]['key'] == df.iloc[i]['key']:
                interval = df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']
                keystroke_intervals.append(interval)
            
            if df.iloc[i]['key'] == 'Backspace' and df.iloc[i]['eventType'] == 'keydown':
                backspace_count += 1
                error_indicators += 1
        
        features.update({
            'key_event_count': len(df),
            'unique_keys': len(df['key'].unique()),
            'backspace_frequency': backspace_count / len(df) if len(df) > 0 else 0,
            'error_rate': error_indicators / len(df) if len(df) > 0 else 0,
            'keystroke_mean_interval': np.mean(keystroke_intervals) if keystroke_intervals else 0,
            'keystroke_std_interval': np.std(keystroke_intervals) if keystroke_intervals else 0,
            'keystroke_max_interval': np.max(keystroke_intervals) if keystroke_intervals else 0,
            'keystroke_min_interval': np.min(keystroke_intervals) if keystroke_intervals else 0,
        })

    # Click patterns
    click_events = data.get('clickEvents', [])
    if len(click_events) >= 2:
        df = pd.DataFrame(click_events)
        df['time_between_clicks'] = df['timestamp'].diff().fillna(0)
        
        features.update({
            'click_count': len(df),
            'click_x_std': df['x'].std(),
            'click_y_std': df['y'].std(),
            'click_mean_interval': df['time_between_clicks'].mean(),
            'click_std_interval': df['time_between_clicks'].std(),
        })
        
    # Scroll behavior
    scroll_events = data.get('scrollEvents', [])
    if len(scroll_events) >= 3:
        df = pd.DataFrame(scroll_events)
        df['timestamp_diff'] = df['timestamp'].diff().fillna(0)
        df['scroll_diff'] = df['scrollY'].diff().fillna(0)
        
        df['scroll_speed'] = np.where(df['timestamp_diff'] > 0, df['scroll_diff'] / df['timestamp_diff'], 0)
        df['scroll_acceleration'] = df['scroll_speed'].diff().fillna(0)
        df['scroll_jerk'] = df['scroll_acceleration'].diff().fillna(0)
        
        features.update({
            'scroll_count': len(df),
            'scroll_mean_speed': df['scroll_speed'].mean(),
            'scroll_std_speed': df['scroll_speed'].std(),
            'scroll_mean_acceleration': df['scroll_acceleration'].mean(),
            'scroll_std_acceleration': df['scroll_acceleration'].std(),
            'scroll_jerkiness': df['scroll_jerk'].abs().mean(),
        })
        
    # Session features
    session_info = data.get('sessionInfo', {})
    duration = session_info.get('duration', 0)
    
    total_events = (
        len(data.get('mouseEvents', [])) +
        len(data.get('keyEvents', [])) +
        len(data.get('clickEvents', [])) +
        len(data.get('scrollEvents', []))
    )
    
    features.update({
        'session_duration': duration,
        'total_event_count': total_events,
        'events_per_second': total_events / (duration / 1000) if duration > 0 else 0,
    })
    
    return features

@app.route('/api/captcha-data', methods=['POST'])
def collect_data():
    data = request.json
    
    # Add metadata
    data['metadata'] = {
        'ip': request.remote_addr,
        'user_agent': request.headers.get('User-Agent'),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save data to file (for data collection)
    filename = f"data/raw/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.remote_addr}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    # If model is loaded, make a prediction
    if model:
        try:
            # Extract features from the data
            features = extract_features(data)
            
            # Create DataFrame and ensure all required features exist
            features_df = pd.DataFrame([features])
            
            # Fill missing features with zeros
            for feature in required_features:
                if feature not in features_df:
                    features_df[feature] = 0
                    
            # Only keep features used by the model
            features_df = features_df[required_features]
            
            # Make prediction
            prediction_proba = model.predict_proba(features_df)[0, 1]  # Probability of being a bot
            prediction = 1 if prediction_proba > 0.5 else 0
            
            result = {
                "status": "success",
                "is_bot": bool(prediction),
                "confidence": float(prediction_proba),
                "challenge_required": bool(prediction_proba > 0.3)
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            # If prediction fails, default to success but flag for review
            result = {
                "status": "success",
                "challenge_required": True,
                "error": str(e)
            }
    else:
        # If no model is loaded, default to success but always show challenge
        result = {
            "status": "success", 
            "challenge_required": True,
            "message": "Running in data collection mode"
        }
    
    return jsonify(result)

@app.route('/api/challenge', methods=['POST'])
def challenge():
    """Endpoint for additional challenges if the user is suspected to be a bot"""
    # This would implement additional verification methods
    data = request.json
    
    # Simple example - check if challenge response matches expected value
    if data.get('challenge_response') == data.get('expected_response'):
        return jsonify({"status": "success", "message": "Verification complete"})
    else:
        return jsonify({"status": "failed", "message": "Verification failed"})

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)