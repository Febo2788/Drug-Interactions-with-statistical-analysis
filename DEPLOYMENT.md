# Deployment Guide

**Author**: Felix Borrego | felix.borrego02@gmail.com

## Quick Deploy

### Local Development
```bash
# Web Interface
python launch_web.py

# API Server
python web_interface/backend/api.py
```

### Docker
```bash
# Single container
docker build -t ddi-predictor .
docker run -p 8501:8501 ddi-predictor

# Full stack
docker-compose up
```

## Production Deployment

### Cloud Platforms
- **Streamlit Cloud**: Deploy directly from GitHub
- **Heroku**: Use included Dockerfile
- **AWS/GCP**: Container deployment ready

### Requirements
- Python 3.9+
- 512MB RAM minimum
- Port 8501 (web) or 8000 (API)

### Environment Variables
```bash
PYTHONPATH=/app/src
STREAMLIT_SERVER_PORT=8501
API_PORT=8000
```

### Health Checks
- Web: `http://localhost:8501`
- API: `http://localhost:8000/health`

Production-ready with CI/CD pipeline included.