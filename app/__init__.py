from flask import Flask
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "app/static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

from app import routes 
