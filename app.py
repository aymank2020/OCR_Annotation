"""
Flask API Server for VideoX Action Recognition
Complete REST API with file upload and processing
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import yaml
import torch
from werkzeug.utils import secure_filename
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from model_architecture import create_model
from inference_module import ActionRecognitionInference

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs/api_predictions'
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global model (loaded once)
model = None
inference_engine = None
config = None
device = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def initialize_model():
    """Initialize model on startup"""
    global model, inference_engine, config, device
    
    try:
        print("=" * 70)
        print("Atlas Action Recognition API Server")
        print("=" * 70)
        print()
        print("=" * 70)
        print("Initializing Action Recognition Model...")
        print("=" * 70)
        
        # Load config
        config = yaml.safe_load(open('config/config.yaml', 'r', encoding='utf-8'))
        
        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
        
        # Create model
        model = create_model(config)
        
        # Load checkpoint
        checkpoint_path = Path(config['training']['checkpoint_dir']) / 'best.pth'
        
        if checkpoint_path.exists():
            inference_engine = ActionRecognitionInference(
                model=model,
                config=config,
                device=device,
                checkpoint_path=str(checkpoint_path)
            )
            print("✅ Model initialized successfully!")
        else:
            print(f"⚠️  WARNING: No checkpoint found at {checkpoint_path}")
            print("   The API will start, but annotation will not work.")
            print("   Please train the model first:")
            print("   python main.py --mode train")
            
            # Create inference engine without checkpoint
            inference_engine = ActionRecognitionInference(
                model=model,
                config=config,
                device=device,
                checkpoint_path=None
            )
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/')
def index():
    """Serve web interface"""
    try:
        return send_file('web_interface.html')
    except:
        return """
        <html>
        <body style="font-family: Arial; padding: 50px; text-align: center;">
            <h1>🎬 Atlas Action Recognition API</h1>
            <p>API Server is running!</p>
            <h2>Available Endpoints:</h2>
            <ul style="list-style: none;">
                <li><a href="/api/status">/api/status</a> - Check server status</li>
                <li>/api/annotate (POST) - Upload and annotate video</li>
                <li>/api/results/{video_id} (GET) - Get prediction results</li>
                <li>/api/download/{video_id}/{format} (GET) - Download results</li>
            </ul>
            <p><em>Note: web_interface.html not found. Use API endpoints directly.</em></p>
        </body>
        </html>
        """


@app.route('/api/status', methods=['GET'])
def status():
    """Get server status"""
    return jsonify({
        'status': 'running',
        'model_loaded': inference_engine is not None,
        'device': device,
        'cuda_available': torch.cuda.is_available(),
        'checkpoint_loaded': inference_engine is not None
    })


@app.route('/api/annotate', methods=['POST'])
def annotate_video():
    """
    Annotate uploaded video
    
    Request:
        - file: video file (multipart/form-data)
        
    Response:
        - JSON with predictions
    """
    try:
        print("\n" + "=" * 70)
        print("ANNOTATION REQUEST")
        print("=" * 70)
        
        # Check model
        if inference_engine is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please train the model first: python main.py --mode train'
            }), 503
        
        print("✅ Model is loaded")
        
        # Check file
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'allowed': list(ALLOWED_EXTENSIONS)
            }), 400
        
        print(f"✅ Received file: {file.filename}")
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, saved_filename)
        
        file.save(filepath)
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✅ File saved: {file_size:.2f} MB")
        
        # Process video
        print("Starting video processing...")
        result = inference_engine.predict_video(filepath)
        
        # Save results
        video_id = Path(filepath).stem
        output_path = Path(OUTPUT_FOLDER) / f"{video_id}.json"
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Predictions saved: {video_id}")
        print(f"   Segments found: {result['num_segments']}")
        print("=" * 70)
        
        return jsonify({
            'success': True,
            'video_id': video_id,
            'filename': filename,
            'duration': result['duration'],
            'num_segments': result['num_segments'],
            'segments': result['segments']
        })
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': 'Processing failed',
            'message': str(e)
        }), 500


@app.route('/api/results/<video_id>', methods=['GET'])
def get_results(video_id):
    """Get prediction results for a video"""
    try:
        result_path = Path(OUTPUT_FOLDER) / f"{video_id}.json"
        
        if not result_path.exists():
            return jsonify({'error': 'Results not found'}), 404
        
        with open(result_path, 'r') as f:
            results = json.load(f)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<video_id>/<format>', methods=['GET'])
def download_results(video_id, format):
    """
    Download results in different formats
    
    Formats: json, atlas, csv
    """
    try:
        result_path = Path(OUTPUT_FOLDER) / f"{video_id}.json"
        
        if not result_path.exists():
            return jsonify({'error': 'Results not found'}), 404
        
        with open(result_path, 'r') as f:
            results = json.load(f)
        
        if format == 'json':
            return send_file(result_path, as_attachment=True)
        
        elif format == 'atlas':
            # Create Atlas format file
            atlas_path = Path(OUTPUT_FOLDER) / f"{video_id}_atlas.txt"
            
            with open(atlas_path, 'w') as f:
                f.write(results['atlas_format'])
            
            return send_file(atlas_path, as_attachment=True)
        
        elif format == 'csv':
            # Create CSV file
            csv_path = Path(OUTPUT_FOLDER) / f"{video_id}.csv"
            
            with open(csv_path, 'w') as f:
                f.write("Start,End,Duration,Action,Confidence\n")
                for seg in results['segments']:
                    f.write(f"{seg['start']},{seg['end']},{seg['duration']},\"{seg['action']}\",{seg['confidence']}\n")
            
            return send_file(csv_path, as_attachment=True)
        
        else:
            return jsonify({'error': 'Invalid format. Use: json, atlas, or csv'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/videos', methods=['GET'])
def list_videos():
    """List all processed videos"""
    try:
        results = []
        
        for result_file in Path(OUTPUT_FOLDER).glob('*.json'):
            with open(result_file, 'r') as f:
                result = json.load(f)
                results.append({
                    'video_id': result['video_id'],
                    'duration': result['duration'],
                    'num_segments': result['num_segments']
                })
        
        return jsonify({'videos': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Initialize model
    print("\n" + "=" * 70)
    print("Starting Flask server...")
    print("=" * 70)
    
    if initialize_model():
        print("\n🌐 Server starting on: http://localhost:5000")
        print("\nAPI Endpoints:")
        print("  📊 Status:     http://localhost:5000/api/status")
        print("  📤 Upload:     POST http://localhost:5000/api/annotate")
        print("  📥 Results:    GET http://localhost:5000/api/results/{video_id}")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 70)
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("\n❌ Failed to initialize model. Server not started.")
        print("Please check the error messages above.")
