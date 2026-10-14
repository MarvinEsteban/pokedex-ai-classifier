# ⚡ Pokédex AI Classifier

Clasificador de Pokémon **Legendarios** con un `RandomForestClassifier` (scikit-learn) y una interfaz web en **Streamlit** lista para desplegar en **Render**.

## Estructura del proyecto

```
├── pokemon.csv            # Dataset (800 Pokémon, 65 legendarios)
├── entrenar.py            # Entrena el modelo y lo guarda en modelo.joblib
├── modelo.joblib          # Modelo entrenado (generado por entrenar.py)
├── app.py                 # Aplicación web Streamlit
├── requirements.txt       # streamlit, scikit-learn, pandas, joblib
├── .streamlit/config.toml # Tema oscuro base
└── .gitignore             # venv/, .env, __pycache__/
```

## Puesta en marcha

```bash
python -m venv venv
venv\Scripts\activate          # En Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python entrenar.py             # Genera modelo.joblib
streamlit run app.py
```

**Render** → Build command: `pip install -r requirements.txt` · Start command:
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
(sube `modelo.joblib` al repositorio para no tener que reentrenar en el despliegue).

---

## 1. `entrenar.py` — Entrenamiento

### Variables
```python
FEATURES = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Generation"]
TARGET = "Legendary"
```
Las 6 estadísticas base más la generación son las variables predictoras (X); `Legendary` (True/False) es la variable objetivo (y). No se usan `Name` ni los tipos.

### División train/test
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
El 80 % (640 filas) se usa para entrenar y el 20 % (160) se reserva para evaluar con datos que el modelo no ha visto. `random_state=42` hace la división reproducible.

### El modelo
```python
RandomForestClassifier(
    n_estimators=100,      # 100 árboles de decisión
    max_features="sqrt",   # cada división solo mira √7 ≈ 2 variables al azar
    max_depth=10,          # profundidad máxima por árbol (evita sobreajuste)
    min_samples_split=5,   # un nodo necesita ≥5 muestras para dividirse
    oob_score=True,        # validación Out-of-Bag
    random_state=42,
)
```
Un **Random Forest** entrena muchos árboles distintos y combina sus votos:
- **Bagging**: cada árbol se entrena con una muestra aleatoria *con reemplazo* del conjunto de entrenamiento (bootstrap).
- **Aleatoriedad en las variables** (`max_features="sqrt"`): en cada división el árbol solo puede elegir entre unas pocas variables, lo que hace que los árboles sean diferentes entre sí y el conjunto generalice mejor.
- **Criterio Gini** (por defecto): cada división se elige para que los grupos resultantes sean lo más "puros" posible (todos legendarios o todos comunes).

### Métricas
| Métrica | Valor | Qué significa |
|---|---|---|
| `oob_score_` | **0.9422** | Precisión calculada con las muestras que *no* entraron en el bootstrap de cada árbol: una validación "gratis" sin usar el test |
| Precisión Test | **0.9625** | Aciertos sobre las 160 filas reservadas |

> Ojo: solo ~8 % de los Pokémon son legendarios, así que un modelo que dijera siempre "Común" ya acertaría un 92 %. La mejora real está en detectar correctamente a los legendarios.

### Persistencia
```python
joblib.dump(modelo, "modelo.joblib")
```
Serializa el bosque entrenado para que la app lo cargue sin reentrenar.

---

## 2. `app.py` — Aplicación Streamlit

### Carga del modelo con caché
```python
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo.joblib")
```
`@st.cache_resource` hace que el modelo se cargue **una sola vez** y se comparta entre todas las sesiones y recargas, en lugar de leerse del disco en cada interacción (Streamlit re-ejecuta el script entero cada vez que se toca un widget).

### Estilos CSS
Se inyecta un bloque `<style>` con `st.markdown(..., unsafe_allow_html=True)`. Define variables de color (`--crimson`, `--gold`…) y estiliza:
- La cabecera tipo Pokédex (lente azul + luces).
- Las tarjetas `.card` con bordes redondeados y sombra.
- El botón con degradado carmesí → dorado y efecto `hover`.
- `.result-legend` (borde dorado con animación de brillo) y `.result-common` (tonos azul/gris).

### Pestaña a) Predicción individual
Seis `st.slider` (estadísticas) y un `st.selectbox` (generación 1–6). Al pulsar el botón:
```python
entrada = pd.DataFrame([valores], columns=FEATURES)
es_legendario = bool(modelo.predict(entrada)[0])
prob_leg = modelo.predict_proba(entrada)[0][IDX_LEGENDARIO]
```
- `predict` → veredicto final (voto mayoritario de los 100 árboles).
- `predict_proba` → fracción de árboles que votan "Legendario"; se muestra en una barra de progreso.
- Se calcula el **Total** sumando las 6 estadísticas.

`IDX_LEGENDARIO = list(modelo.classes_).index(True)` asegura que se lee la columna correcta de `predict_proba`.

#### Insignias de confianza del bosque
El veredicto principal **siempre** es el de `predict` (mayoría de votos, umbral 50 %). Debajo se añade información complementaria:

| Condición | Insignia | Significado |
|---|---|---|
| `prob_leg > 0.70` o `prob_leg < 0.35` | ✔ Alta confianza del bosque | La gran mayoría de árboles coincide |
| `0.35 ≤ prob_leg ≤ 0.70` | 🔥 Zona de duda · Élite | Los árboles están divididos |
| Total de stats `== 600` | 💎 Total 600 · perfil pseudo-legendario | Curiosidad: los pseudolegendarios de los juegos suman exactamente 600 |

La zona de duda no se llama "pseudo-legendario" a propósito: en el dataset, de los 37 Pokémon que caen entre 35 % y 70 %, **30 son legendarios reales** (Articuno, Zapdos, Entei, Regirock…), mientras que los pseudolegendarios reales reciben probabilidades muy bajas (Dragonite 9 %, Tyranitar 4 %). Por eso la insignia de Total 600 va aparte y puede aparecer junto a cualquier veredicto.

| Ejemplo | Veredicto | Confianza | Total 600 |
|---|---|---|---|
| Mewtwo | Legendario | Alta | — |
| Articuno | Legendario | Zona de duda | — |
| Dragonite | Común | Alta | 💎 |

### Pestaña b) Inferencia masiva (CSV)
`st.file_uploader` recibe un CSV; se comprueba que estén las 7 columnas necesarias, se convierten a numérico (las filas no válidas se marcan en lugar de romper la app) y se añaden las columnas `Predicción` y `Prob. Legendario`. El resultado se muestra con `st.dataframe` y se puede descargar.

### Pestaña c) Métricas del bosque
Lee los hiperparámetros directamente del modelo cargado (`n_estimators`, `criterion`, `oob_score_`, `max_depth`…) y los muestra con `st.metric`. `st.bar_chart` dibuja `feature_importances_`.

---

## 3. ¿Qué decide si un Pokémon es Legendario o Común?

### Importancia de las variables
| Variable | Importancia |
|---|---|
| Sp. Atk | 0.223 |
| Speed | 0.200 |
| HP | 0.161 |
| Sp. Def | 0.149 |
| Attack | 0.115 |
| Defense | 0.113 |
| Generation | 0.040 |

La importancia mide cuánto reduce la impureza Gini cada variable, sumado en todos los árboles.

### La clave: el poder total
Comparando el dataset:

| | Común (media) | Legendario (media) |
|---|---|---|
| HP | 67 | 93 |
| Attack | 76 | 117 |
| Defense | 72 | 100 |
| **Sp. Atk** | 68 | **122** |
| Sp. Def | 69 | 106 |
| **Speed** | 65 | **100** |
| **Total** | **417** | **637** |

- **Ningún legendario tiene un Total por debajo de 580.** Por debajo de ~540 el modelo prácticamente nunca dice "Legendario".
- Las mayores diferencias están en **Ataque Especial** (+54) y **Velocidad** (+35): los legendarios suelen ser rápidos y con mucho poder especial, por eso son las variables más importantes.
- Además, los legendarios tienen estadísticas **altas y equilibradas** (mínimo 50 en HP, Speed y Sp. Atk), mientras que muchos comunes destacan en una sola estadística.
- **Generation** apenas influye: hay legendarios en todas las generaciones.

Prueba con todas las estadísticas iguales (generación 3):

| Cada stat | Total | Prob. Legendario |
|---|---|---|
| 80 | 480 | 1 % |
| 90 | 540 | 56 % |
| 100 | 600 | 55 % |
| 110 | 660 | 79 % |
| 120 | 720 | 86 % |

### La zona difícil (580–700)
48 Pokémon **comunes** también superan los 580 puntos: las **Mega Evoluciones** (Mega Charizard, Mega Gyarados…) y los **pseudolegendarios** (Dragonite, Tyranitar, Salamence, Hydreigon). Como solo vemos las estadísticas, el modelo no siempre puede distinguirlos y es donde se concentran sus errores (por ejemplo, confunde Mega Salamence o Hydreigon con legendarios, y Regirock o Uxie con comunes).

**Resumen:** el modelo aprende que un legendario es un Pokémon con estadísticas totales muy altas (≥ ~580), sobre todo en Ataque Especial y Velocidad, y repartidas de forma equilibrada. Dentro de esa zona de élite, separar legendarios de Megas y pseudolegendarios es lo que más le cuesta.
