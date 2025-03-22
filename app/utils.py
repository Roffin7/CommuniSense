import os
import random
import subprocess
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
import torchaudio
import torch
import cv2
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
from ultralytics import YOLO

torchaudio.set_audio_backend("sox_io")

asr_model_name = "facebook/wav2vec2-large-960h"
asr_processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_name)

emotion_model = pipeline("audio-classification", model="facebook/wav2vec2-large-xlsr-53")
yolo_model = YOLO("yolov8n.pt")

HAND_GESTURE_LABELS = {
    0: "No hands visible",
    1: "Open palm",
    2: "Pointing",
    3: "Thumbs up",
    4: "Crossed arms",
    5: "Fidgeting"
}

def extract_audio(video_path, audio_path):
    subprocess.run(["ffmpeg", "-i", video_path, audio_path, "-y"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return os.path.exists(audio_path)

def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.replace(".mp3", ".wav")
    if not os.path.exists(wav_path):
        subprocess.run(["ffmpeg", "-i", mp3_path, wav_path, "-y"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return wav_path if os.path.exists(wav_path) else None

def transcribe_audio(audio_path):
    if audio_path.endswith(".mp3"):
        audio_path = convert_mp3_to_wav(audio_path)
        if not audio_path:
            return "Error: Failed to convert MP3 to WAV.", None

    try:
        waveform, sample_rate = sf.read(audio_path, dtype="float32")
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)

        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            waveform = resample_poly(waveform, target_sample_rate, sample_rate)

        input_values = asr_processor(waveform, sampling_rate=target_sample_rate, return_tensors="pt").input_values
        with torch.no_grad():
            logits = asr_model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = asr_processor.decode(predicted_ids[0])

        return transcript, logits

    except Exception as e:
        print(f"⚠️ Transcription Error: {e}")
        return "Error processing audio", None

def calculate_speaking_pace(audio_path, transcript):
    try:
        with sf.SoundFile(audio_path) as audio_file:
            duration = len(audio_file) / audio_file.samplerate

        if duration == 0:
            return 0

        word_count = len(transcript.split())
        words_per_minute = (word_count / duration) * 60
        print(f"🥳{words_per_minute}🥳")
        return int(words_per_minute)

    except Exception as e:
        print(f"⚠️ Speaking Pace Calculation Error: {e}")
        return None

def analyze_body_language(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    gesture_count = {gesture: 0 for gesture in HAND_GESTURE_LABELS.values()}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        results = yolo_model(frame)

        if results:
            detected_objects = results[0].boxes

            for box in detected_objects:
                class_id = int(box.cls[0])
                
                if class_id in HAND_GESTURE_LABELS:
                    gesture = HAND_GESTURE_LABELS[class_id]
                    gesture_count[gesture] += 1

        if frame_count >= 30:
            break

    cap.release()

    most_common_gesture = max(gesture_count, key=gesture_count.get)

    if most_common_gesture in ["Open palm", "Thumbs up"]:
        return " Your hand gestures made your speech more engaging and confident!"
    elif most_common_gesture == "Pointing":
        return " Your gestures added emphasis, but be mindful of overusing them to maintain a natural flow."
    elif most_common_gesture == "Crossed arms":
        return " Your posture seemed closed-off. Try to keep an open stance for a more welcoming presence."
    elif most_common_gesture == "Fidgeting":
        return " There were signs of nervousness in your gestures. Relax and maintain steady movements for a composed delivery."
    else:
        return " Try using more hand gestures and facial expressions to make your speech feel more dynamic!"

def analyze_emotion(audio_path):
    try:
        results = emotion_model(audio_path)
        if not results:
            return "Error analyzing emotions"
        
        top_emotion = results[0]["label"]
        confidence = results[0]["score"]

        emotion_mapping = {
            "LABEL_0": "Neutral",
            "LABEL_1": "Happy",
            "LABEL_2": "Sad",
            "LABEL_3": "Angry",
            "LABEL_4": "Fearful"
        }

        if confidence < 0.5:
            return "Unclear Emotion"

        return emotion_mapping.get(top_emotion, "Neutral")
    except Exception as e:
        print(f"⚠️ Emotion Analysis Error: {e}")
        return "Error analyzing emotions"

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    face_detected_frames = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        results = yolo_model(frame)

        if results and len(results[0].boxes) > 0:
            face_detected_frames += 1

        if face_detected_frames >= 5:
            cap.release()
            return "Good face detected"

    cap.release()
    return "Face not visible - Please ensure your face is clearly seen"

def calculate_clarity_score(transcript, logits):
    if transcript == "Error processing audio" or logits is None:
        return 0

    confidence_scores = torch.softmax(logits, dim=-1).max(dim=-1).values.mean().item()
    words = transcript.split()

    if confidence_scores < 0.6 or len(words) < 5:  
        return max(min(len(words) * 2, 20) * confidence_scores, 10)  

    length_score = min(len(words) * 5, 100)
    adjusted_score = length_score * confidence_scores
    return int(min(adjusted_score, 100))

def calculate_speaking_score(clarity_score, emotion):
    emotion_scores = {
        "Neutral": 70,
        "Happy": 90,
        "Sad": 50,
        "Angry": 40,
        "Fearful": 45,
        "Unclear Emotion": 30
    }
    emotion_score = emotion_scores.get(emotion, 60)

    if clarity_score < 40:
        emotion_score *= 0.5  

    return int((clarity_score * 0.6) + (emotion_score * 0.4))

def generate_feedback(clarity_score, speaking_score, face_visibility, body_language_feedback):
    if face_visibility != "Good face detected":
        return ("⚠️ It looks like your face wasn't clearly visible. To make a strong impact, ensure good lighting, "
                "face the camera directly, and remove any obstructions. Your audience connects better when they see you!")

    if clarity_score < 30 or speaking_score < 40:
        return ("⚠️ Your speech was not very clear. Try to slow down, pronounce words distinctly, and reduce background noise. "
                "Speaking at a steady pace and practicing articulation will greatly improve your delivery!")

    feedback_options = {
        "excellent": [
            "⭐ Fantastic work! Your speech was clear, engaging, and confident. Keep up this great energy{}!",
            "🎉 Amazing performance! Your confidence and clarity made your speech highly effective{}. Keep refining your delivery!"
        ],
        "good": [
            "👍 Good job! Try to add a bit more variation in your tone to keep your audience engaged{}. ",
            "✅ Well done! Focus on smoother transitions between ideas to improve the natural flow of your speech{}."
        ],
        "average": [
            "📢 You're on the right track! Work on articulating your words more clearly for better impact{}. ",
            "🎤 Decent effort! Try to reduce pauses and maintain a steady rhythm to sound more confident{}."
        ],
        "needs_improvement": [
            "⚠️ Your speech could be clearer. Slow down and pronounce your words distinctly to improve comprehension{}.",
            "📌 Pay attention to articulation and pronunciation small adjustments will make a big difference{}!",
            "🛑 Your speech was hard to understand at times. Focus on a steady pace and clearer pronunciation{}."
        ]
    }

    if clarity_score > 85 and speaking_score > 85:
        feedback_category = "excellent"
    elif clarity_score > 65 and speaking_score > 65:
        feedback_category = "good"
    elif clarity_score > 45 and speaking_score > 45:
        feedback_category = "average"
    else:
        feedback_category = "needs_improvement"

    final_feedback = random.choice(feedback_options[feedback_category]).format(body_language_feedback)
    return final_feedback

print("✅ Models Loaded Successfully")
