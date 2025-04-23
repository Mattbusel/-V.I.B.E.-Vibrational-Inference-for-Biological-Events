import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


def generate_synthetic_vibration_data(duration=100, fs=1000, mutation_probability=0.01):
    """
    Simulate vibration data with rare mutation events.
    """
    time = np.linspace(0, duration, duration * fs)
    vibration_data = np.sin(2 * np.pi * 10 * time)  
    mutation_indices = np.random.choice(time.shape[0], size=int(mutation_probability * len(time)), replace=False)

    
    for idx in mutation_indices:
        vibration_data[idx:idx+10] += np.random.normal(10, 5, 10)

    return time, vibration_data


def preprocess_vibration_data(vibration_data, fs=1000):
    """
    Preprocess the vibration data by applying FFT to extract frequency-domain features.
    """
   
    N = len(vibration_data)
    frequency = np.fft.fftfreq(N, d=1/fs)
    fft_values = fft(vibration_data)
    
   
    return np.abs(fft_values)


time, vibration_data = generate_synthetic_vibration_data()
fft_features = preprocess_vibration_data(vibration_data)


labels = np.zeros(len(vibration_data))
labels[np.random.choice(len(vibration_data), size=int(0.01 * len(vibration_data)), replace=False)] = 1


features = fft_features.reshape(-1, 1)  
scaler = StandardScaler()  
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


plt.plot(time, vibration_data)
plt.title("Vibration Signal with Mutations")
plt.xlabel("Time (s)")
plt.ylabel("Vibration Amplitude")
plt.show()


plt.plot(np.fft.fftfreq(len(vibration_data), d=1/1000), fft_features)
plt.title('Frequency Domain Representation of Vibration Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()


import joblib
joblib.dump(model, 'vibe_model.pkl')
