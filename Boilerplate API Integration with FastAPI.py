from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

model = joblib.load('vibe_model.pkl')


app = FastAPI()


class VibrationData(BaseModel):
    vibration_signal: list

def preprocess_vibration_data(vibration_data, fs=1000):
    N = len(vibration_data)
    fft_values = fft(vibration_data)
    return np.abs(fft_values).reshape(-1, 1)


@app.post("/predict/")
async def predict_mutation(vibration_data: VibrationData):
    
    try:
        vibration_signal = vibration_data.vibration_signal
        if len(vibration_signal) == 0:
            raise ValueError("Vibration signal cannot be empty.")

     
        features = preprocess_vibration_data(vibration_signal)

      
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

       
        prediction = model.predict(features_scaled)

        result = {
            "prediction": int(prediction[0]),
            "description": "Mutation detected" if prediction[0] == 1 else "No mutation detected"
        }
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")


@app.post("/visualize/")
async def visualize_vibration(vibration_data: VibrationData):
    try:
        vibration_signal = vibration_data.vibration_signal
        if len(vibration_signal) == 0:
            raise ValueError("Vibration signal cannot be empty.")
        
        
        plt.plot(vibration_signal)
        plt.title("Vibration Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        
        plt.savefig("vibration_signal.png")
        return {"message": "Visualization saved", "file": "vibration_signal.png"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in visualization: {str(e)}")


@app.get("/health/")
async def health_check():
    return {"status": "Model is running smoothly"}
