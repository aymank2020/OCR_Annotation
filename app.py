"""
Flask API Server for VideoX Action Recognition
Complete REST API with file upload and processing
DEMO MODE: Works without trained model for testing
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
import cv2
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

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
DEMO_MODE = True  # Start in demo mode


def allowed_file(filename):
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def get_video_duration(filepath):
    """Get video duration in seconds using OpenCV"""
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration


def generate_demo_annotation(duration, num_segments=3):
    """Generate demo annotations based on video duration"""

    # Sample actions from Egocentric Annotation Program
    demo_actions = [
        "pick up mat",
        "place mat on table",
        "move mat to table",
        "adjust mat position",
        "pick up assembly parts",
        "place assembled parts",
        "smooth on table",
        "fold shirt",
        "place pen in box",
        "pick up screwdriver"
    ]

    # Generate segments
    segment_duration = duration / num_segments

    segments = []
    for i in range(num_segments):
        start = i * segment_duration
        end = (i + 1) * segment_duration
        action = random.choice(demo_actions)
        confidence = round(random.uniform(0.85, 0.98), 2)

        segments.append({
            'start': round(start, 1),
            'end': round(end, 1),
            'duration': round(end - start, 1),
            'action': action,
            'confidence': confidence
        })

    # Generate Atlas format
    atlas_lines = []
    for i, seg in enumerate(segments, 1):
        start_min = int(seg['start'] // 60)
        start_sec = int(seg['start'] % 60)
        end_min = int(seg['end'] // 60)
        end_sec = int(seg['end'] % 60)
        action_cap = seg['action'].capitalize()
        atlas_lines.append(
            f"{start_min}:{start_sec:02d}.{int((seg['start'] % 1) * 10):01d}-{end_min}:{end_sec:02d}.{int((seg['end'] % 1) * 10):01d}#{i} {action_cap}"
        )

    return segments, '\n'.join(atlas_lines)


def initialize_model():
    """Initialize model on startup (will fail gracefully in demo mode)"""
    global model, inference_engine, config, device, DEMO_MODE

    try:
        print("=" * 70)
        print("Atlas Action Recognition API Server")
        print("=" * 70)
        print()

        # Try to load config
        try:
            config = yaml.safe_load(open('config/config.yaml', 'r', encoding='utf-8'))
        except:
            config = None

        # Check for GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
        print(f"CUDA Available: {torch.cuda.is_available()}")

        # Try to import and create model
        try:
            from model_architecture import create_model
            from inference_module import ActionRecognitionInference

            model = create_model(config)

            # Check for checkpoint
            checkpoint_path = Path('outputs/checkpoints/best.pth')
            if checkpoint_path.exists():
                inference_engine = ActionRecognitionInference(
                    model=model,
                    config=config,
                    device=device,
                    checkpoint_path=str(checkpoint_path)
                )
                print("✅ Model initialized successfully!")
                DEMO_MODE = False
            else:
                print("⚠️  No checkpoint found - running in DEMO MODE")
                print(f"   Looking for: outputs/checkpoints/best.pth")
                print()
                print("   DEMO MODE FEATURES:")
                print("   • Video upload and processing works")
                print("   • Demo annotations generated for testing")
                print("   • Full annotation validation available")
                print("   • No GPU/Model training required")
                print()
                print("   To use real model:")
                print("   1. Train the model locally: python main.py --mode train")
                print("   2. Upload best.pth to outputs/checkpoints/")
                print("   3. Restart the server")
                DEMO_MODE = True

        except ImportError as e:
            print(f"⚠️  Could not import model modules: {e}")
            print("   Running in DEMO MODE")
            DEMO_MODE = True

        print()
        return True

    except Exception as e:
        print(f"⚠️  Error during initialization: {e}")
        print("   Running in DEMO MODE")
        DEMO_MODE = True
        return True


@app.route('/')
def index():
    """Serve web interface"""
    try:
        return send_file('web_interface.html')
    except:
        return """
        <html>
        <head>
            <title>Atlas Action Recognition - Demo Mode</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 50px;
                    text-align: center;
                    color: white;
                }
                h1 { font-size: 3em; margin-bottom: 20px; }
                .status { background: rgba(255,255,255,0.2); padding: 20px;
                          border-radius: 10px; margin: 30px auto; max-width: 600px; }
                .endpoint { background: rgba(0,0,0,0.3); padding: 15px;
                           margin: 10px; border-radius: 5px; }
                a { color: #ffd700; }
            </style>
        </head>
        <body>
            <h1>🎬 Atlas Action Recognition</h1>
            <div class="status">
                <h2>🎯 Mode: DEMO MODE</h2>
                <p>API Server is running without trained model</p>
                <p>Trial annotations generated for testing</p>
            </div>
            <h2>Available Endpoints:</h2>
            <div class="endpoint">
                <a href="/api/status">/api/status</a> - Check server status
            </div>
            <div class="endpoint">POST /api/annotate - Upload and annotate video</div>
            <div class="endpoint">GET /api/results/{video_id} - Get prediction results</div>
            <div class="endpoint">GET /api/download/{video_id}/{format} - Download results</div>
            <p style="margin-top: 30px;"><em>Note: web_interface.html should be present. Use API endpoints directly.</em></p>
        </body>
        </html>
        """


@app.route('/api/status', methods=['GET'])
def status():
    """Get server status"""
    return jsonify({
        'status': 'running',
        'demo_mode': DEMO_MODE,
        'model_loaded': inference_engine is not None,
        'device': device,
        'cuda_available': torch.cuda.is_available(),
        'checkpoint_loaded': inference_engine is not None,
        'message': 'Demo Mode - Generating test annotations' if DEMO_MODE else 'Model Mode - Using trained model'
    })


@app.route('/api/annotate', methods=['POST'])
def annotate_video():
    """
    Annotate uploaded video (Demo Mode or Real Model)

    Request:
        - file: video file (multipart/form-data)

    Response:
        - JSON with predictions
    """
    try:
        print("\n" + "=" * 70)
        print("ANNOTATION REQUEST")
        print("=" * 70)

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

        # Get video duration
        duration = get_video_duration(filepath)
        print(f"✅ Video duration: {duration:.1f} seconds")

        # Process video (Demo or Real)
        if DEMO_MODE:
            print("🎯 Processing in DEMO MODE (generating test annotations)")
            segments, atlas_format = generate_demo_annotation(duration)
            result = {
                'video_id': Path(filepath).stem,
                'filename': filename,
                'duration': duration,
                'num_segments': len(segments),
                'segments': segments,
                'atlas_format': atlas_format,
                'demo_mode': True,
                'message': 'Demo annotations for testing interface'
            }
        else:
            if inference_engine is None:
                return jsonify({
                    'error': 'Model not loaded in non-demo mode',
                    'message': 'Please check model initialization'
                }), 503

            print("🤖 Processing with trained model")
            result = inference_engine.predict_video(filepath)
            result['demo_mode'] = False

        # Save results
        video_id = Path(filepath).stem
        output_path = Path(OUTPUT_FOLDER) / f"{video_id}.json"

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"✅ Results saved: {video_id}")
        print(f"   Segments: {result['num_segments']}")
        print("=" * 70)

        return jsonify({
            'success': True,
            'demo_mode': DEMO_MODE,
            'video_id': result['video_id'],
            'filename': result.get('filename', filename),
            'duration': result['duration'],
            'num_segments': result['num_segments'],
            'segments': result['segments'],
            'message': 'Demo annotations' if DEMO_MODE else 'Model predictions'
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
                    'num_segments': result['num_segments'],
                    'filename': result.get('filename', result['video_id'])
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
        print("\nMode:", "🎯 DEMO MODE" if DEMO_MODE else "🤖 MODEL MODE")
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