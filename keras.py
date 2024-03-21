import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
import matplotlib.pyplot as plt


data = pd.read_csv("input/data/Tesla_Nasdaq_Prediction.csv")

features = data.iloc[:, 2:].values
target = data['Close/Last'].values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

model = Sequential([
    SimpleRNN(256, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=True),
    Dropout(0.25),
    SimpleRNN(128, return_sequences=True),
    Dropout(0.25),
    SimpleRNN(64),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(
    X_train_reshaped, y_train_scaled, epochs=50, batch_size=32, validation_split=0.1, verbose=1
)

preds_scaled = model.predict(X_test_reshaped)
preds = scaler.inverse_transform(preds_scaled)

plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual Value', color='red', linewidth=2)
plt.plot(preds, label='Predicted Value', color='blue', linewidth=2)
plt.title('TESLA/USD Predictions')
plt.xlabel('Sample Index')
plt.ylabel('TESLA/USD Value')
plt.legend()
plt.show()
plt.savefig('output/fig/TESLA_USD_Predictions.png')
plt.clf()  # Clear the current figure

mae_score = mean_absolute_error(y_test, preds)
mse_score = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"MAE Score: {mae_score}")
print(f"MSE Score: {mse_score}")
print(f"R2 Score: {r2 * 100}%")

plt.plot(history.history['loss'], label='Training loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation loss', linewidth=2)
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('output/fig/Training_validation_loss.png')
plt.clf()  # Clear the current figure

model.save("output/model/trained_model.h5")
print("Model saved")
