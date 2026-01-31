import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



from pprint import pprint
import time

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from scipy import stats

HEADER_MARKER = "n"  
# Tokens típicos que representan missing
MISSING_TOKENS = {
    "", "NA", "N/A", "NAN", "NULL", "NONE", "-", "--", ".", "S/I",
    "SIN DATO", "Sin dato", "sin dato"
}

# arriba del archivo (junto a MISSING_TOKENS)
MISSING_TOKENS_UPPER = {t.upper() for t in MISSING_TOKENS}

def normalize_missing(s: pd.Series, missing_tokens=MISSING_TOKENS) -> pd.Series:
    x = s.astype("string").str.strip()
    # usa cache precomputada si no te pasaron otros tokens
    if missing_tokens is MISSING_TOKENS:
        mask = x.str.upper().isin(MISSING_TOKENS_UPPER)
    else:
        mask = x.str.upper().isin({t.upper() for t in missing_tokens})
    return x.mask(mask, pd.NA)


def load_csv_repair_headers(
    path: str,
    header_marker: str = HEADER_MARKER,
    read_csv_kwargs: dict | None = None
) -> pd.DataFrame:
    """
    Carga CSV con headers repetidos:
    - Lee sin header (header=None)
    - Detecta fila de encabezado donde col0 == header_marker
    - Asigna columnas desde esa fila
    - Elimina filas de encabezado repetido dentro del cuerpo
    """
    read_csv_kwargs = read_csv_kwargs or {}

    # 1) Carga cruda sin header
    df_raw = pd.read_csv(
        path,
        header=None,
        dtype="string",          # ayuda a detectar marker sin problemas de tipos
        encoding_errors="replace",
        on_bad_lines="skip",
        **read_csv_kwargs
    )

    # 2) Detecta candidatos de header (col 0 == marker)
    col0 = df_raw.iloc[:, 0].astype("string").str.strip()
    header_candidates = df_raw.index[col0.eq(str(header_marker))].tolist()

    # 3) Si no hay marker, fallback a lectura normal
    if not header_candidates:
        df = pd.read_csv(
            path,
            encoding_errors="replace",
            on_bad_lines="skip",
            **read_csv_kwargs
        )
        return df.reset_index(drop=True)

    # 4) Usa el primer header encontrado
    header_row = header_candidates[0]
    header = df_raw.iloc[header_row].astype("string").str.strip().tolist()

    df = df_raw.iloc[header_row + 1:].copy()
    df.columns = header
    df = df.reset_index(drop=True)

    # 5) Elimina headers repetidos dentro del cuerpo
    col0_body = df.iloc[:, 0].astype("string").str.strip()
    df = df.loc[~col0_body.eq(str(header_marker))].copy()

    return df.reset_index(drop=True)

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    - strip
    - reemplaza espacios múltiples por uno
    - convierte None/NA en nombres seguros
    - resuelve duplicados con sufijos _2, _3, ...
    """
    cols = pd.Series(df.columns, dtype="string").fillna(pd.NA)
    cols = cols.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)

    # Reemplaza vacíos/NA
    cols = cols.mask(cols.isna() | (cols == ""), "col_sin_nombre")

    # Resolver duplicados
    seen = {}
    new_cols = []
    for c in cols.tolist():
        if c not in seen:
            seen[c] = 1
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")

    df = df.copy()
    df.columns = new_cols
    return df

def to_int64_robust(s: pd.Series) -> pd.Series:
    x = normalize_missing(s)
    # Quita separadores de miles comunes y deja solo dígitos y signo
    x = (x.str.replace(r"[,\s]", "", regex=True)
           .str.replace(r"[^0-9\-]", "", regex=True))

    out = pd.to_numeric(x, errors="coerce")
    return out.astype("Int64")

def to_float_robust(s: pd.Series) -> pd.Series:
    x = normalize_missing(s)
    x = x.str.replace(r"\s", "", regex=True)

    # NUEVO: si viene en formato ES "1.234,56" -> "1234,56" -> "1234.56"
    both = x.str.contains(r"\.") & x.str.contains(r",")
    x = x.mask(both, x.where(~both).fillna(x).str.replace(".", "", regex=False))

    x = x.str.replace(",", ".", regex=False)
    x = x.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(x, errors="coerce")


def to_binary_int64(s: pd.Series) -> pd.Series:
    x = normalize_missing(s)

    # Map de valores típicos
    map_bin = {
        "SI": 1, "SÍ": 1, "YES": 1, "Y": 1,
        "NO": 0, "N": 0,
        "TRUE": 1, "FALSE": 0,
        "1": 1, "0": 0
    }

    # Normaliza a mayúsculas para mapear
    x_up = x.str.upper()

    # Aplica mapping y luego numérico
    mapped = x_up.map(map_bin)
    # Si ya venía como "0/1" numérico en string, intenta convertir
    fallback_num = pd.to_numeric(x, errors="coerce")

    out = mapped.where(mapped.notna(), fallback_num)
    out = out.round()
    out = out.where(out.isin([0, 1]), np.nan)
    return out.astype("Int64")

def to_datetime_robust(s: pd.Series) -> pd.Series:
    x = normalize_missing(s)
    return pd.to_datetime(x, errors="coerce", dayfirst=True)



def apply_typing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in cols_int:
        if c in df.columns:
            df[c] = to_int64_robust(df[c])

    for c in cols_float:
        if c in df.columns:
            df[c] = to_float_robust(df[c])

    for c in cols_bin:
        if c in df.columns:
            df[c] = to_binary_int64(df[c])

    for c in cols_date:
        if c in df.columns:
            df[c] = to_datetime_robust(df[c])

    for c in cols_id_string:
        if c in df.columns:
            df[c] = normalize_missing(df[c])  # mantiene strings consistentes + NA

    return df

cols_int = [
    "edad","n_notas_parciales","ind_nsp","n_nsp","nmero_de_asignaturas_inscritas",
    "_de_asistencia_del_periodo","_asignaturas_con_nsp","n_tutoras","n_cuotas_total",
    "n_cuotas_morosas","dias_mora","q_reclamos_internos","cantidad_de_ingresos_torniquetes","saldo_total","saldo_moroso"
]

cols_float = [
    "avance_de_malla","progreso_acadmico","promedio_notas_acumulado","promedio_notas_parciales"
    
]

cols_bin = [
    "alumno_sies","plan_abierto_en_srm","_logro_diagnsticos_de_ingreso",
    "_diagnosticos_de_ingreso_rendidos","otras_titulaciones_o_egresos_udla",
    "otras_carreras_abandonadas_udla","indicador_cae","indicador_beca","cursa_mentoras",
    "_talleres_acompaamiento_aprobados","reclamo_ses","reclamo_sernac","indicador_de_gratuidad",
]

cols_date = [
    "fecha_de_baja","fecha_ltimo_requerimiento_web","ultima_fecha_acceso_torniquetes",
    "ltimo_inicio_sesin_a_miudla_independiente_del_periodo"
]

cols_id_string = [
    "rut_alumno","pidm","email","nombre","periodo_acadmico",
    "periodo_de_ingreso_a_la_institucin","periodo_de_ingreso_a_la_carrera","gnero"
]

def make_risk_label_from_vigencia(v: pd.Series) -> pd.Series:
    x = normalize_missing(v).astype("string").str.upper()
    return (x.str.contains("NO") & x.str.contains("VIGENTE")).astype(int)

# ============================================================
# 1) Definir TARGET coherente: riesgo = 1 si NO vigente
#    (ajusta aquí si tu vigencia viene como texto)
# ============================================================

def build_target_riesgo(df, col_vigencia="vigencia"):
    """
    Retorna y (riesgo=1 abandono / no vigente).
    - Si vigencia es 0/1: riesgo = (vigencia==0)
    - Si vigencia es string: intenta mapear valores comunes.
    """
    s = df[col_vigencia]

    if pd.api.types.is_numeric_dtype(s):
        # asumimos vigencia 1 = vigente, 0 = no vigente
        return (s.astype("Int64") == 0).astype(int)

    # Caso string: normalizar
    s2 = s.astype(str).str.upper().str.strip()

    # map común: "VIGENTE" / "NO VIGENTE" / "RETENIDO" / "DESERTA" etc.
    vig_true = {"VIGENTE", "SI", "SÍ", "ACTIVE", "ACTIVO", "RETENIDO", "RETENCION", "RETENCIÓN", "1"}
    vig_false = {"NO VIGENTE", "NO", "INACTIVO", "DESERTA", "DESERTO", "ABANDONA", "0"}

    # si cae en vig_false => riesgo=1; si cae en vig_true => riesgo=0
    riesgo = np.where(s2.isin(vig_false), 1,
             np.where(s2.isin(vig_true), 0, np.nan))

    y = pd.Series(riesgo, index=df.index, name="riesgo").astype("float")
    # Si hay nans, los dejamos fuera (no inventamos)
    return y

#============================================================
# 2) Selección de features + dropear columnas no predictivas
# ============================================================
def drop_id_like_cols(df):
    drop_candidates = [
        "rut_alumno", "pidm", "nombre", "email",
        "fecha_de_baja", "tipo_de_baja", "n",  # ajusta si corresponde
    ]
    drop_candidates = [c for c in drop_candidates if c in df.columns]
    return df.drop(columns=drop_candidates, errors="ignore")

# ============================================================
# 3) Split: preferencia HOLDOUT TEMPORAL si existe periodo_acadmico
# ============================================================
def temporal_holdout_split(dfX, y, col_periodo="periodo_acadmico"):
    """
    Si existe periodo_acadmico:
      - test = max(periodo)
      - train = resto
    Si no existe, retorna None.
    """
    if col_periodo not in dfX.columns:
        return None

    # asegurar numérico comparable
    per = pd.to_numeric(dfX[col_periodo], errors="coerce")
    if per.isna().all():
        return None

    last_period = int(np.nanmax(per))
    mask_test = (per == last_period)

    X_train = dfX.loc[~mask_test].copy()
    y_train = y.loc[~mask_test].copy()
    X_test  = dfX.loc[mask_test].copy()
    y_test  = y.loc[mask_test].copy()

    

    return X_train, X_test, y_train, y_test, last_period




# ============================================================
# 4) Construir preprocessors UNA SOLA VEZ desde X_train
# ============================================================
def build_preprocessors(X_train):
    # Detectar columnas por dtype
    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocess LINEAL: imputación + scaler
    preprocess_linear = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False))
            ]), num_features),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_features),
        ],
        remainder="drop"
    )

    # Preprocess TREE: imputación + ohe (sin scaler)
    preprocess_tree = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
            ]), num_features),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_features),
        ],
        remainder="drop"
    )

    return preprocess_linear, preprocess_tree, num_features, cat_features


# ============================================================
# 5) Métricas + evaluación única y coherente (riesgo=1)
# ============================================================
def get_score_or_none(model, X):
    """
    Retorna score continuo para ROC/PR si soporta:
    - predict_proba -> proba clase 1
    - decision_function -> score
    Si no, None.
    """
    # Pipeline o estimator
    final_est = model.steps[-1][1] if isinstance(model, Pipeline) else model

    if hasattr(final_est, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            return proba[:, 1]
        except Exception:
            return None

    if hasattr(final_est, "decision_function"):
        try:
            scores = model.decision_function(X)
            return np.asarray(scores).ravel()
        except Exception:
            return None

    return None


def eval_model(name, model, X_test, y_test, thr=0.5):
    """
    Evalúa modelo para riesgo=1.
    thr aplica si hay score continuo; si no, usa predict().
    """
    y_score = get_score_or_none(model, X_test)

    if y_score is not None:
        y_pred = (y_score >= thr).astype(int)
        roc = roc_auc_score(y_test, y_score)
        pr  = average_precision_score(y_test, y_score)
    else:
        # fallback solo si realmente no hay score
        y_pred = model.predict(X_test)
        roc, pr = np.nan, np.nan

    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, digits=3)

    return {
        "Modelo": name,
        "ROC_AUC(riesgo)": roc,
        "PR_AUC(riesgo)": pr,
        "F1(riesgo)": f1,
        "thr": thr,
        "cm": cm,
        "report": rep
    }

# ============================================================
# 6) Curvas ROC y PR para todos los clasificadores válidos
# ============================================================
def plot_roc_curves(all_models, X_test, y_test):
    plt.figure(figsize=(7, 6))
    skipped = []

    for name, model in all_models.items():
        y_score = get_score_or_none(model, X_test)
        if y_score is None:
            skipped.append(name)
            continue
        auc_val = roc_auc_score(y_test, y_score)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curvas ROC — comparación de modelos (riesgo = 1)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if skipped:
        print("Modelos omitidos en ROC (sin score de clasificación):", skipped)


def plot_pr_curves(all_models, X_test, y_test):
    plt.figure(figsize=(7, 6))
    skipped = []

    for name, model in all_models.items():
        y_score = get_score_or_none(model, X_test)
        if y_score is None:
            skipped.append(name)
            continue
        ap = average_precision_score(y_test, y_score)
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curvas Precision–Recall — comparación de modelos (riesgo = 1)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if skipped:
        print("Modelos omitidos en PR (sin score de clasificación):", skipped)

# ============================================================
# 7) Tuning DecisionTree: Random Search amplio -> Grid enfocado
# ============================================================
def tune_dt_random_and_grid(pipe_dt, X_train, y_train, cv5, scoring="average_precision", random_state=42):
    # Paso 1: Random Search amplio
    param_dist_dt_wide = {
        "model__max_depth": [None, 3, 5, 8, 12, 16, 20],
        "model__min_samples_split": stats.randint(2, 51),
        "model__min_samples_leaf": stats.randint(1, 26),
        "model__criterion": ["gini", "entropy", "log_loss"]
    }

    rand_dt_wide = RandomizedSearchCV(
        estimator=pipe_dt,
        param_distributions=param_dist_dt_wide,
        n_iter=40,
        cv=cv5,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        return_train_score=False,
        verbose=1
    )

    print("Paso 1: Random Search amplio...")
    t0 = time.perf_counter()
    rand_dt_wide.fit(X_train, y_train)
    rand_wide_time = time.perf_counter() - t0

    best_params = rand_dt_wide.best_params_
    print(f"Tiempo Random amplio: {rand_wide_time:.2f}s")
    print("Mejores params (Random amplio):")
    pprint(best_params)

    # Paso 2: Grid enfocado alrededor del mejor punto
    best_depth = best_params.get("model__max_depth", None)
    if best_depth is None:
        depth_candidates = [None, 3, 5]
    else:
        depth_candidates = [max(1, best_depth - 2), best_depth, best_depth + 2]

    param_grid_dt_focused = {
        "model__max_depth": list(dict.fromkeys(depth_candidates)),
        "model__min_samples_split": sorted(list(set([
            max(2, best_params["model__min_samples_split"] - 5),
            best_params["model__min_samples_split"],
            best_params["model__min_samples_split"] + 5
        ]))),
        "model__min_samples_leaf": sorted(list(set([
            max(1, best_params["model__min_samples_leaf"] - 3),
            best_params["model__min_samples_leaf"],
            best_params["model__min_samples_leaf"] + 3
        ]))),
        "model__criterion": [best_params["model__criterion"]]
    }

    print("\nPaso 2: Grid Search enfocado...")
    pprint(param_grid_dt_focused)

    grid_dt_focused = GridSearchCV(
        estimator=pipe_dt,
        param_grid=param_grid_dt_focused,
        cv=cv5,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        verbose=1
    )

    t0 = time.perf_counter()
    grid_dt_focused.fit(X_train, y_train)
    grid_focused_time = time.perf_counter() - t0

    print(f"\nTiempo Grid enfocado: {grid_focused_time:.2f}s")
    print("Mejores params (Grid enfocado):")
    pprint(grid_dt_focused.best_params_)
    print(f"Mejor PR-AUC (Grid enfocado): {grid_dt_focused.best_score_:.4f}")

    return rand_dt_wide, rand_wide_time, grid_dt_focused, grid_focused_time, param_grid_dt_focused
