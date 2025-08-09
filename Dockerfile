FROM python:3.11-slim

WORKDIR /app

# Install Node.js and npm
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy GUI source and build it
COPY ode_gui_bundle/package*.json ./ode_gui_bundle/
WORKDIR /app/ode_gui_bundle
RUN npm install

COPY ode_gui_bundle/ ./
RUN npm run build && \
    echo "Contents of dist after build:" && \
    ls -la dist/

# Copy the rest of the application
WORKDIR /app
COPY . .

# Verify GUI was built
RUN echo "Checking GUI build..." && \
    ls -la ode_gui_bundle/ && \
    ls -la ode_gui_bundle/dist/ || echo "dist directory not found!"

EXPOSE 8080

CMD ["python", "scripts/production_server.py"]
