import os

class Config:
    SECRET_KEY = "your_secret_key"
    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    ALLOWED_EXTENSIONS = {"mp4", "avi", "mov"}

    DB_HOST = "localhost"
    DB_USER = "your_username"
    DB_PASSWORD = "your_password"
    DB_NAME = "your_database"
    DB_PORT = 3306

def configure_app(app):
    app.config.from_object(Config)
