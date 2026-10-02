import io
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import app as module


@pytest.fixture
def client(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    outputs = tmp_path / "outputs"
    uploads.mkdir()
    outputs.mkdir()
    monkeypatch.setattr(module, "UPLOAD_FOLDER", str(uploads))
    monkeypatch.setattr(module, "OUTPUT_FOLDER", str(outputs))
    monkeypatch.setattr(module, "DEMO_MODE", True)
    module.app.config.update(TESTING=True, MAX_CONTENT_LENGTH=module.MAX_FILE_SIZE)
    yield module.app.test_client()
    module.app.config['MAX_CONTENT_LENGTH'] = module.MAX_FILE_SIZE


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def upload(client, video):
    return client.post('/api/annotate', data={'video': (io.BytesIO(video), 'fixture.avi')})


def test_real_video_upload_results_and_download_roundtrip(client, video):
    response = upload(client, video)
    assert response.status_code == 200
    result = response.json
    assert result['duration'] == pytest.approx(1.0)
    assert result['demo_mode'] is True
    assert result['num_segments'] == 3
    stored = client.get(f"/api/results/{result['video_id']}")
    assert stored.status_code == 200
    assert stored.json['segments'] == result['segments']
    assert client.get(f"/api/download/{result['video_id']}/json").status_code == 200


def test_same_filename_uploads_have_distinct_saved_results(client, video):
    first = upload(client, video).json
    second = upload(client, video).json
    assert first['video_id'] != second['video_id']
    assert len(client.get('/api/videos').json['videos']) == 2


def test_unreadable_upload_is_rejected_without_retaining_input(client):
    response = client.post('/api/annotate', data={'video': (io.BytesIO(b'invalid'), 'invalid.mp4')})
    assert response.status_code == 400
    assert not list(Path(module.UPLOAD_FOLDER).iterdir())
    assert not list(Path(module.OUTPUT_FOLDER).iterdir())


def test_oversized_upload_returns_413(client):
    module.app.config['MAX_CONTENT_LENGTH'] = 100
    response = client.post('/api/annotate', data={'video': (io.BytesIO(b'x' * 1000), 'large.mp4')})
    assert response.status_code == 413
    assert not list(Path(module.UPLOAD_FOLDER).iterdir())


def test_model_output_receives_persisted_video_identity(client, video, monkeypatch):
    monkeypatch.setattr(module, 'DEMO_MODE', False)
    monkeypatch.setattr(module, 'inference_engine', SimpleNamespace(predict_video=lambda path: {'duration': 1.0, 'num_segments': 0, 'segments': [], 'atlas_format': ''}))
    response = upload(client, video)
    assert response.status_code == 200
    assert client.get(f"/api/results/{response.json['video_id']}").json['video_id'] == response.json['video_id']


def test_demo_status_works_without_torch(client, monkeypatch):
    monkeypatch.setattr(module, 'torch', None)
    result = client.get('/api/status')
    assert result.status_code == 200
    assert result.json['cuda_available'] is False
