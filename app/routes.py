import os
from flask import render_template, request, jsonify
from werkzeug.utils import secure_filename
from app import app
from app.utils import (
    extract_audio, transcribe_audio, analyze_emotion, analyze_video,
    calculate_clarity_score, calculate_speaking_score, analyze_body_language,
    generate_feedback, calculate_speaking_pace
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/record")
def analyze_vdo():
    return render_template("record.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["video"]
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(video_path)

    audio_path = os.path.splitext(video_path)[0] + ".mp3"
    extract_audio(video_path, audio_path)

    transcript, logits = transcribe_audio(audio_path)
    emotion = analyze_emotion(audio_path)
    face_result = analyze_video(video_path)
    clarity_score = calculate_clarity_score(transcript, logits)
    speaking_score = calculate_speaking_score(clarity_score, emotion)
    speaking_pace = calculate_speaking_pace(audio_path, transcript)  
    body_language_feedback = analyze_body_language(video_path)

    final_feedback = generate_feedback(clarity_score, speaking_score, face_result, body_language_feedback)

    return jsonify({
        "face_detection": face_result,
        "voice_clarity_score": clarity_score,
        "overall_speaking_score": speaking_score,
        "speaking_pace_wpm": speaking_pace, 
        "body_language_feedback": body_language_feedback,
        "final_feedback": final_feedback
    })

@app.route("/list_files", methods=["GET"])
def list_files():
    """Lists all existing uploaded files."""
    files = os.listdir(app.config["UPLOAD_FOLDER"])
    video_files = [f for f in files if f.endswith((".mp4", ".avi", ".mov"))]  # Adjust extensions if needed
    return jsonify(video_files)

@app.route("/analyze_existing", methods=["POST"])
def analyze_existing():
    """Analyzes an existing video file from the uploads folder."""
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "No file selected"}), 400

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    audio_path = os.path.splitext(video_path)[0] + ".mp3"

    extract_audio(video_path, audio_path)

    transcript, logits = transcribe_audio(audio_path)
    emotion = analyze_emotion(audio_path)
    face_result = analyze_video(video_path)
    clarity_score = calculate_clarity_score(transcript, logits)
    speaking_score = calculate_speaking_score(clarity_score, emotion)
    speaking_pace = calculate_speaking_pace(audio_path, transcript)
    body_language_feedback = analyze_body_language(video_path)

    final_feedback = generate_feedback(clarity_score, speaking_score, face_result, body_language_feedback)

    return jsonify({
        "face_detection": face_result,
        "voice_clarity_score": clarity_score,
        "overall_speaking_score": speaking_score,
        "speaking_pace_wpm": speaking_pace,
        "body_language_feedback": body_language_feedback,
        "final_feedback": final_feedback
    })
