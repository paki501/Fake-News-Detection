from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
import pickle
import re
import nltk
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import uvicorn

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load trained model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()


# Mount static files (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Template rendering
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Request body model
class NewsRequest(BaseModel):
    text: str

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]  # Lemmatization
    return " ".join(words)

# @app.post("/predict")
# def predict_news(request: NewsRequest):
#     # Preprocess input text
#     clean_text = preprocess_text(request.text)

#     # Transform using the same vectorizer
#     text_transformed = vectorizer.transform([clean_text])

#     # Predict using trained model
#     prediction = model.predict(text_transformed)[0]

#     return {"prediction": "Fake News" if prediction == 1 else "Real News"}


@app.post("/predict/")
async def predict_news(request: Request, news_text: str = Form(...)):
    try:
        # Preprocess input text
        text_transformed = vectorizer.transform([news_text])

        # Make prediction
        prediction = model.predict(text_transformed)

        # Convert prediction to readable output
        result = "Real" if prediction[0] == 1 else "Fake"

        return templates.TemplateResponse("index.html", {"request": request, "prediction": result})
    
    except Exception as e:
        return {"error": str(e)}



# ✅ Run the app with Uvicorn (only if script is run directly)
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
