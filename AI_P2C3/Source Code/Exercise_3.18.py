# Penyelesaian Exercise 3.18
from auto_ml import Predictor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

# Memuat data
california = fetch_california_housing(as_frame=True)
df = california.frame

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

column_descriptions = {
    'MedHouseVal': 'output' # Kolom target untuk prediksi harga
}

ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)
ml_predictor.train(df_train)
ml_predictor.score(df_test, df_test.MedHouseVal)
