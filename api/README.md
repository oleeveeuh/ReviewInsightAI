# ReviewInsight Agent API

Prototype FastAPI service exposing the plan-and-execute review analyzer.

## Starting the server

```bash
# from project root (core requirements are enough to start)
uvicorn api.main:app --port 8000
```

The service starts and reports health **without** an OpenAI API key; the
`POST /analyze` endpoint requires the key (and `requirements-optional.txt`)
at request time.

## Documentation

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Endpoints

| Endpoint | Method | Needs API key | Purpose |
|---|---|---|---|
| `/analyze` | POST | at request time | Plan-and-execute analysis of one review |
| `/memory/stats` | GET | no | Aggregates from the local analysis log |
| `/tools` | GET | no | Registered agent tools |
| `/health` | GET | no | Service health + whether an LLM key is configured |

## Example usage

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Great benefits but too much overtime", "review_id": "test_001"}'

curl http://127.0.0.1:8000/health
```

## Operational notes

- **CORS**: configured via `ALLOWED_ORIGINS` (comma-separated origins).
  Unset means no cross-origin browser access. Wildcards are rejected.
- **Errors**: client-visible error bodies are generic; details are logged
  server-side only.
- **Limits**: review text is capped at 10,000 characters (422 on violation).
- Bind to a specific host and run behind a real ASGI deployment if you ever
  expose this beyond localhost; this is a prototype, not a hardened service.
- Output "retention risk" is an LLM-generated classification of review text,
  not a validated prediction about any person.
