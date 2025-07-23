from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from model import heart_model, HeartRiskModel
from sklearn.preprocessing import MinMaxScaler
import io
import pandas as pd
import traceback
import numpy as np

app = FastAPI()
model = HeartRiskModel()

# Папка с шаблонами
templates = Jinja2Templates(directory=".")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/", response_class=HTMLResponse)

async def handle_form(
    request: Request,
    age: float = Form(...),
    gender: str = Form(...),
    heart_rate: float = Form(...),
    cholesterol: float = Form(...),
    sleep_hours_per_day: float = Form(...),
    sedentary_hours_per_day: float = Form(...),
    income: float = Form(...),
    bmi: float = Form(...),
    triglycerides: float = Form(...),
    physical_activity_days_per_week: float = Form(...),
    exercise_hours_per_week: float = Form(...),
    systolic_blood_pressure: float = Form(...),
    diastolic_blood_pressure: float = Form(...),
    
):
    # Собираем признаки в нужном порядке
    features = [[
        age,
        cholesterol,
        heart_rate,
        exercise_hours_per_week,
        sedentary_hours_per_day,
        income,
        bmi,
        triglycerides,
        sleep_hours_per_day,
        gender,
        systolic_blood_pressure,
        diastolic_blood_pressure,
        physical_activity_days_per_week, 
        
        
    ]]
    
    columns = [
        "age",
        "cholesterol",
        "heart_rate",
        "exercise_hours_per_week",
        "sedentary_hours_per_day",
        "income",
        "bmi",
        "triglycerides",
        "sleep_hours_per_day",
        "gender",
        "systolic_blood_pressure",
        "diastolic_blood_pressure",
        "physical_activity_days_per_week",  
    ]

    df = pd.DataFrame(features, columns=columns)
    
    probabilities = heart_model.predict(df)
    risk = float(np.round(probabilities[0] * 100, 2))
    prediction = 'Да' if probabilities[0] >= 0.2 else 'Нет'
    return templates.TemplateResponse("index.html", {"request": request, "result": risk, 'answer': prediction})


@app.post("/predict/")
async def predict_from_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV.")

    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))

        probabilities, predictions, idxs = model.predict(df)

        results = []
        for idx, i in enumerate(idxs):
            results.append({
                "id": i,
                "probability": round(probabilities[idx] * 100, 2),
                "prediction": int(predictions[idx])
            })

        return JSONResponse(content={"results": results})

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None,
        "answer": None})

@app.get("/form", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})