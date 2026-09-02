# MGC Sales Assistant

A lightweight FastAPI app that lets a user ask questions about property documents and receive answers from either an n8n workflow or a built-in local fallback search.

## Overview

This project includes:

- a simple HTML front end for asking sales questions
- a FastAPI backend for processing requests
- optional n8n integration for AI-powered responses
- local document retrieval fallback when the external service is unavailable

## Project Structure

```text
main.py                  FastAPI app entry point
 doc_assistant.py        AI answer logic with n8n + local fallback
 lead_scorer.py          Lead scoring model and utility logic
 templates/index.html    Main user interface
 documents/              Sample sales and policy documents
 requirements.txt        Python dependencies
 README.md               Project documentation
```

## Tech Stack

- Python 3.10+
- FastAPI
- Jinja2 Templates
- Uvicorn
- Optional n8n webhook integration

## Installation

```bash
cd mgc_fastapi
pip install -r requirements.txt
```

## Run the app

```bash
uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## n8n Integration

The app can use an external n8n agent for document answers. To enable it, create a webhook workflow in n8n and set the URL in your environment:

### Windows PowerShell

```powershell
$env:N8N_WEBHOOK_URL = "https://your-webhook-url"
```

### macOS / Linux

```bash
export N8N_WEBHOOK_URL="https://your-webhook-url"
```

If the webhook is unavailable, the application falls back to local document search using the files in the `documents/` folder.

## How it works

- The user submits a question from the homepage.
- The backend sends the question to the configured answer source.
- The response is returned to the frontend and displayed in the page.
- If no external service is configured or it fails, the app still answers using local project documents.

## Notes

- Replace sample files in `documents/` with your real materials for better results.
- The app is intentionally simple and lightweight for internal use or quick demos.
