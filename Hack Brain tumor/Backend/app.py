from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import io
from PIL import Image

app = FastAPI()

# Allow all CORS origins for simplicity (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy model for testing - replace with your real model
class DummyModel:
    def __call__(self, x):
        return torch.tensor([[0.4, 0.6]])  # Always predicts 'Tumor' with 60% confidence

model = DummyModel()

@app.get("/")
def home():
    return {"status": "running", "message": "Upload to /predict endpoint"}
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")

        # Preprocess the image (resize, normalize, tensor, batch dim)
        x = preprocess_image(img).to(device)

        model.eval()  # Make sure model is in eval mode

        with torch.no_grad():
            logits = model(x)  # Run the model
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # Get softmax probabilities

        confidence = probs[1]

        # Decide prediction based on confidence thresholds
        if confidence > 0.75:
            prediction = "Tumor"
        elif confidence > 0.4:
            prediction = "Uncertain"
        else:
            prediction = "No Tumor"

        return JSONResponse({
            "prediction": prediction,
            "probabilities": probs.tolist()  # convert numpy array to list for JSON
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return JSONResponse({"error": str(e)}, status_code=400)
