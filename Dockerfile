# Use a pre-configured image that already has dlib and face_recognition installed
FROM animcogn/face_recognition:latest

# Set the working directory
WORKDIR /app

# Copy your local code into the container
COPY . .

# Install the lighter requirements (OpenCV and Flask/Django)
RUN pip install --no-cache-dir opencv-python-headless Flask numpy

# Start your application (Replace 'app.py' with your main file name)
CMD ["python", "app.py"]