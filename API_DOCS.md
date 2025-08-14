# API Documentation

**Author**: Felix Borrego | felix.borrego02@gmail.com

## Quick Start
```bash
python web_interface/backend/api.py
# Visit: http://localhost:8000/docs
```

## Core Endpoints

### Prediction
- **POST /predict** - Single drug pair prediction
- **POST /batch** - Batch prediction (max 100 pairs)

### Data Access
- **GET /drugs** - List available drugs
- **GET /drug/{name}** - Drug molecular properties
- **GET /interactions** - Known interactions database

### System
- **GET /health** - Health check
- **GET /model/performance** - Model metrics

## Request Examples

```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict", 
                        json={"drug1": "Warfarin", "drug2": "Aspirin"})

# Batch prediction  
batch = {"drug_pairs": [{"drug1": "Drug1", "drug2": "Drug2"}]}
response = requests.post("http://localhost:8000/batch", json=batch)
```

## Response Format
```json
{
  "interaction_probability": 0.95,
  "confidence": 0.85,
  "risk_level": "HIGH",
  "timestamp": "2025-01-15T10:30:00"
}
```

Interactive documentation available at `/docs` endpoint.