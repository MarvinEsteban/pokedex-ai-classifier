"""Entrena un RandomForestClassifier para predecir si un Pokémon es Legendario."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CSV_PATH = "pokemon.csv"
MODEL_PATH = "modelo.joblib"
FEATURES = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Generation"]
TARGET = "Legendary"


def main():
    # 1. Carga de datos
    df = pd.read_csv(CSV_PATH)
    X = df[FEATURES]
    y = df[TARGET].astype(bool)

    # 2. División train/test 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Entrenamiento del bosque aleatorio
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        max_depth=10,
        min_samples_split=5,
        oob_score=True,
        random_state=42,
    )
    modelo.fit(X_train, y_train)

    # 4. Evaluación
    test_acc = accuracy_score(y_test, modelo.predict(X_test))
    print("=" * 50)
    print(" RANDOM FOREST - Clasificador de Pokémon Legendarios")
    print("=" * 50)
    print(f"Muestras train: {len(X_train)} | test: {len(X_test)}")
    print(f"Precisión OOB  (oob_score_): {modelo.oob_score_:.4f}")
    print(f"Precisión Test (accuracy)  : {test_acc:.4f}")
    print("-" * 50)
    print("Importancia de las variables (feature_importances_):")
    importancias = pd.Series(modelo.feature_importances_, index=FEATURES)
    for nombre, valor in importancias.sort_values(ascending=False).items():
        print(f"  {nombre:<12} {valor:.4f}  {'#' * int(valor * 50)}")
    print("=" * 50)

    # 5. Persistencia del modelo
    joblib.dump(modelo, MODEL_PATH)
    print(f"Modelo guardado en '{MODEL_PATH}'")


if __name__ == "__main__":
    main()
