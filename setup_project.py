import os

# Define folder structure
folders = [
    "app/static",
    "app/templates",
    "app/static/uploads",
    "venv"
]

# Define files
files = [
    "app/__init__.py",
    "app/routes.py",
    "app/utils.py",
    "app/models.py",
    "app/config.py",
    "app/templates/index.html",
    "requirements.txt",
    "run.py",
    "README.md"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create empty files
for file in files:
    with open(file, "w") as f:
        f.write("")  # Create an empty file

print("✅ Project structure created successfully!")
