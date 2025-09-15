import os
import re
import json
from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
from pdfplumber import open as pdf_open
from pytesseract import image_to_string
from PIL import Image
import uvicorn

#Sample call
# curl -X POST -F "file=@C:045_Agentic/sample_form.png" http://127.0.0.1:8000/analyze

app = FastAPI()

# Initialize NER pipeline (can swap with fine-tuned CDT model)
ner = pipeline("ner", model="dslim/bert-base-NER")

def extract_text(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        with pdf_open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif ext in [".png", ".jpg", ".jpeg"]:
        return image_to_string(Image.open(file_path))
    else:
        raise ValueError("Unsupported file type")

def extract_insurance_data(text):
    print(f"Extracting data from text of {text}")
    entities = ner(text)
    print(f"NER entities: {entities}")
    cdt_codes = re.findall(r"\bD\d{4}\b", text)
    print(f"CDT codes found: {cdt_codes}")
    downgrades = re.findall(r"Downgraded to (D\d{4})", text)
    exclusions = re.findall(r"Not Covered[:\s]+(D\d{4})", text)
    waiting_periods = re.findall(r"Waiting Period[:\s]+(\d+ months)", text)

    flagged_entities = [e for e in entities if "code" in e["word"].lower() or "coverage" in e["word"].lower()]

    return {
        "cdt_codes": list(set(cdt_codes)),
        "downgraded_codes": list(set(downgrades)),
        "excluded_codes": list(set(exclusions)),
        "waiting_periods": waiting_periods,
        "entities": flagged_entities
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        text = extract_text(temp_path)
        results = extract_insurance_data(text)
    finally:
        os.remove(temp_path)

    return {"status": "success", "results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
