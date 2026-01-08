import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================
# Cargar objetos
# ======================
modelo = joblib.load("modelo_autos.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")
encoder = joblib.load("encoder.pkl")
columnas_modelo = joblib.load("columnas_modelo.pkl")

# ======================
# Interfaz
# ======================
st.title("🚗 Predicción de Precio de Autos")

st.write("Introduce las características del auto:")

# ----------------------
# Variables numéricas
# ----------------------
levy = st.number_input("Levy", min_value=0.0, value=500.0)
mileage = st.number_input("Kilometraje", min_value=0.0, value=50000.0)
engine_volume = st.number_input("Volumen del motor (L)", min_value=0.5, max_value=8.0, value=2.0)
doors = st.selectbox("Número de puertas", [2, 3, 4, 5])
cylinders = st.number_input("Cilindros", min_value=2, max_value=16, value=4)

# ----------------------
# Variables categóricas
# ----------------------
manufacturer = st.selectbox("Marca", encoder.categories_[0])
model = st.selectbox("Modelo", encoder.categories_[1])
prod_year = st.selectbox("Año de producción", encoder.categories_[2])
category = st.selectbox("Categoría", encoder.categories_[3])
leather = st.selectbox("Interior de cuero", encoder.categories_[4])
fuel = st.selectbox("Tipo de combustible", encoder.categories_[5])
gear = st.selectbox("Tipo de transmisión", encoder.categories_[6])
drive = st.selectbox("Tracción", encoder.categories_[7])
wheel = st.selectbox("Volante", encoder.categories_[8])
color = st.selectbox("Color", encoder.categories_[9])
turbo = st.selectbox("Turbo", encoder.categories_[10])

# ======================
# Predicción
# ======================
if st.button("Predecir precio"):
    
    # ----------------------
    # Parte numérica
    # ----------------------
    X_base = pd.DataFrame(
        np.zeros((1, len(columnas_modelo))),
        columns=columnas_modelo
    )

    X_base.loc[0, 'Levy'] = levy
    X_base.loc[0, 'Mileage'] = mileage
    X_base.loc[0, 'Engine volume'] = engine_volume
    X_base.loc[0, 'Doors'] = doors
    X_base.loc[0, 'Cylinders'] = cylinders

    # ----------------------
    # Parte categórica
    # ----------------------
    X_cat = pd.DataFrame([[
        manufacturer, model, prod_year, category,
        leather, fuel, gear, drive, wheel, color, turbo
    ]], columns=encoder.feature_names_in_)

    X_cat_encoded = encoder.transform(X_cat)
    X_cat_encoded = pd.DataFrame(
        X_cat_encoded,
        columns=encoder.get_feature_names_out(),
        index=X_base.index
    )

    # Activar dummies
    X_base[X_cat_encoded.columns] = X_cat_encoded

    # ----------------------
    # Escalado
    # ----------------------
    X_scaled = scaler_X.transform(X_base)

    # ----------------------
    # Predicción
    # ----------------------
    log_price_pred = modelo.predict(X_scaled)
    precio = np.exp(log_price_pred)[0]


    st.success(f"💰 Precio estimado: ${precio:,.2f}")

