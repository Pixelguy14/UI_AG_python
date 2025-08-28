import os

class Config:
    SECRET_KEY = os.urandom(24)
    SESSION_TYPE = 'filesystem'
    SESSION_FILE_DIR = './flask_session_cache'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 800 * 1024 * 1024 * 1024
