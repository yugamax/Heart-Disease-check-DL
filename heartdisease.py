import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler

try:
    df = pd.read_csv(r"heart.csv")
    x = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    over = RandomOverSampler()
    x, y = over.fit_resample(x, y)

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.5, random_state=0)
    x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.6, random_state=0)

    model_path = 'heart_disease_model.keras'
    if os.path.exists(model_path):
        print("Loading saved model")
        yugah = tf.keras.models.load_model(model_path)
    else:
        print("Training new model")
        yugah = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        yugah.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          loss="binary_crossentropy", metrics=["accuracy"])
        yugah.fit(x_train, y_train, batch_size=16, epochs=100, validation_data=(x_valid, y_valid))
        yugah.save(model_path)

    yp = yugah.predict(x_test)
    yp = (yp > 0.5).astype(int)
    acc = accuracy_score(y_test, yp)
    print(f"Model accuracy: {acc:.2f}")
except Exception as e:
    print(f"An error occurred: {str(e)}")

def hd(a):
    l=list(a.keys())
    l=[a[i] for i in l]
    ui = np.array([l])
    ui = scaler.transform(ui)
    res = yugah.predict(ui)[0][0]
    res = (res > 0.5).astype(int)
    if res == 0:
        return "\nYou don't have Heart disease\n"
    else:
        return "\nYou have Heart disease\n"

age = int(input("\nEnter your age : "))
gender = int(input("\nEnter your gender (0: Female , 1: Male) : "))
cp = float(input("\nEnter your chest pain type (1-4) : "))
tb = float(input("\nEnter your resting blood pressure : "))
ch = float(input("\nEnter your serum cholestrol in mg/dl : "))
recg = float(input("\nEnter your resting ECG result : "))
thlch = float(input("\nEnter your maximum heart rate : "))
ex = int(input("\nEnter your excercise induced angina : "))
op = float(input("\nEnter your oldpeak : "))
slope = int(input("\nEnter your slope : "))
mv = int(input("\nEnter your number of major vessels (1-4) : "))

print("\n",hd({"age": age,"gender": gender,"cp": cp,"trestbps": tb,"chol": ch,"restecg": recg,"thalach": thlch,"exang": ex,"oldpeak": op,"slope": slope,"ca": mv}))
