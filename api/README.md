# ReviewInsight Agent API

## Starting the Server
```bash
# From project root
uvicorn api.main:app --reload
```

Server will run at: http://localhost:8000

## Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Example Usage

### Analyze a Review
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Great benefits but too much overtime",
    "review_id": "test_001"
  }'
```

### Get Memory Stats
```bash
curl http://localhost:8000/memory/stats
```

### List Tools
```bash
curl http://localhost:8000/tools
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Response Format

All endpoints return JSON with:
- Structured data models
- Clear error messages
- HTTP status codes (200, 400, 500, 503)

## Features

- ✅ Auto-generated OpenAPI docs
- ✅ CORS enabled for frontend
- ✅ Pydantic validation
- ✅ Error handling
- ✅ Agent initialization on startup
