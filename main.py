from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import doc_assistant
import lead_scorer

app = FastAPI(title="MGC Sales Assistant")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"lead_fields": lead_scorer.get_fields()}
    )


@app.post("/ask")
async def ask(question: str = Form(...)):
    question = question.strip()
    if not question:
        return JSONResponse({"error": "Enter a question."}, status_code=400)
    return JSONResponse(await doc_assistant.answer_question(question))


@app.post("/score")
async def score(request: Request):
    form = await request.form()
    lead = dict(form)
    return JSONResponse(lead_scorer.score_lead(lead))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
