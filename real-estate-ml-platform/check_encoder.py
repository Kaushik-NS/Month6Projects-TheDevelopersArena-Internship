import joblib

encoders = joblib.load("models/encoders.pkl")

print("CITY:")
print(encoders["city"].classes_)

print("\nTYPE:")
print(encoders["type"].classes_)

print("\nNEIGHBORHOOD SAMPLE:")
print(encoders["neighborhood"].classes_[:20])