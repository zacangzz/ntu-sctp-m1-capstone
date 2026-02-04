# Use the official lightweight Python image.
FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the logs.
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install Streamlit and other dependencies.
RUN pip install -r requirements.txt

# Expose the port your app will run on. Cloud Run expects the app to listen on the port defined by the PORT environment variable, which defaults to 8080.
EXPOSE 8080

# Run the service on container startup.
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
