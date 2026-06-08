import joblib

encoders = joblib.load("models/encoders.pkl")

for n in sorted(encoders["neighborhood"].classes_):
    if "siru" in str(n).lower():
        print(n)