# This base image already has the heavy dlib and face_recognition installed!
FROM animcogn/face_recognition:latest

# Set the working directory
WORKDIR /app

# Copy your requirements and install the light web stuff
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code
COPY . .

# Run the app
CMD ["python", "app.py"]