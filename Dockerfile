FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-gui.txt* ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# If you have requirements-gui.txt, install it too
# RUN if [ -f requirements-gui.txt ]; then pip install -r requirements-gui.txt; fi

# Copy and build GUI
COPY ode_gui_bundle/package*.json ./ode_gui_bundle/
RUN cd ode_gui_bundle && npm install

COPY ode_gui_bundle ./ode_gui_bundle
RUN cd ode_gui_bundle && npm run build

# Copy the rest of the application
COPY . .

# Expose port (Railway sets PORT env variable)
EXPOSE 8080

# Start application
CMD ["python", "scripts/production_server.py"]
