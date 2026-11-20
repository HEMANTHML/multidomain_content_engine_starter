# Multi-Domain AI Content Generation Engine

An LLM-powered content generation platform built with Python, Flask, and Groq. It uses a configurable multi-agent workflow to generate, evaluate, and refine content across multiple domains.

## Supported Domains

- Financial Scam Awareness
- Educational Explainers
- Technical Documentation
- Local Business Content

## Features

- YAML-based domain configuration
- Reusable five-agent generation pipeline
- Domain-specific prompts, tone, and constraints
- Automated quality checks using Pydantic
- AI-powered content revision with retry limits
- REST API for domain selection and content generation
- Automated testing and evaluation

## Tech Stack

Python · Flask · Groq · Pydantic · YAML · Pytest · Docker

## Project Structure

```text
├── app.py
├── agents.py
├── profiles.py
├── domains/
│   ├── financial_scam_awareness.yaml
│   ├── educational_explainers.yaml
│   ├── technical_documentation.yaml
│   └── local_business_content.yaml
├── tests/
├── evaluation/
├── requirements.txt
├── Dockerfile
└── README.md
```

## Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/multidomain-content-engine.git
cd multidomain-content-engine

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file and configure your API key:

```env
GROQ_API_KEY=your_api_key
```

Run the application:

```bash
python app.py
```

## API

**Get available domains**
```http
GET /domains
```

**Generate content**
```http
POST /generate
```

Example request:

```json
{
  "domain": "financial_scam_awareness",
  "topic": "How to identify fake banking messages"
}
```

## Testing

```bash
pytest -v
```

## Future Improvements

- Real-time generation updates using SSE
- Generation history and analytics
- Additional domains and LLM providers
- Improved content evaluation

---

Built as a modular, extensible multi-agent AI content generation project.
