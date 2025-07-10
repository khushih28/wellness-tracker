FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make port 8501 available (Streamlit default)
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "wellness_tracker.py"]