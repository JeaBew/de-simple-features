from cassis import load_cas_from_xmi
from py_lift.utils.core import load_lift_typesystem
from py_lift.dkpro import T_FEATURE
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import keep_trailing_number_filename
import shutil



typesystem = load_lift_typesystem()
FEATURE_NAME = "Readability_Score_FleschReadingEase_de"
T_FEATURE = T_FEATURE


# CAS-Datei laden
with open("datasets_lifted/deplain/simple/deplain_simple_117.xmi", "rb") as f:
    cas = load_cas_from_xmi(f, typesystem=typesystem)

# Durch alle Annotationen eines bestimmten Typs iterieren
# (der Typname muss zum LiFT-Schema passen, z.B. etwas wie:)
for annotation in cas.select(T_FEATURE):
    if annotation["name"] == "Readability_Score_FleschReadingEase_de":
        score = annotation["value"]
        print(score)


def plot_deplain():

    ordner_simple = Path("datasets_lifted/deplain/simple/")
    ordner_text = Path("datasets_lifted/deplain/text/")
    deplain_simple = []
    deplain_text = []
    for xmi_file in ordner_simple.glob("*.xmi"):
        print(xmi_file)
        cas = load_cas_from_xmi(xmi_file, typesystem=typesystem)

        for annotation in cas.select(T_FEATURE):
            if annotation["name"] == "Readability_Score_FleschReadingEase_de":
                score = annotation["value"]
                print(score)
                deplain_simple.append(score)

    for xmi_file in ordner_text.glob("*.xmi"):
        print(xmi_file)
        cas = load_cas_from_xmi(xmi_file, typesystem=typesystem)

        for annotation in cas.select(T_FEATURE):
            if annotation["name"] == "Readability_Score_FleschReadingEase_de":
                score = annotation["value"]
                print(score)
                deplain_text.append(score)

    x = np.array(deplain_simple, dtype=float)  # simple
    y = np.array(deplain_text, dtype=float)  # original/text
    assert len(x) == len(y)

    plt.scatter(x, y, s=20, alpha=0.7)
    plt.xlabel("simple")
    plt.ylabel("text")
    plt.title("Punktwolke: text gegen simple")
    plt.grid(True, alpha=0.3)

    # Referenzlinie y = x (mittig/diagonal)
    mn = min(x.min(), y.min())
    mx = max(x.max(), y.max())
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=2, label=r"Referenz: $y=x$")

    # Optional: Achsen gleich skalieren, damit die Diagonale wirklich 45° ist
    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.legend()
    plt.show()


def extract_feature_value(filepath: Path, typesystem, feature_type: str, feature_name: str):
    """
    Lädt eine einzelne .xmi-Datei und liest den Wert eines bestimmten
    Features (über den generischen name/value-Annotationstyp) aus.
    Gibt None zurück, falls das Feature nicht gefunden wird.
    """
    with open(filepath, "rb") as f:
        cas = load_cas_from_xmi(f, typesystem=typesystem)

    for annotation in cas.select(feature_type):
        if annotation["name"] == feature_name:
            try:
                return float(annotation["value"])
            except (TypeError, ValueError):
                return None
    return None

def extract_id_from_filename(filename: str, prefix: str = None) -> str:
    """Nutzt einfach den Dateinamen (ohne .xmi-Endung) als ID."""
    return filename.replace(".xmi", "")


def build_readability_dataframe(
        original_dir: str,
        simplified_dir: str,
        typesystem_path: str,
        feature_type: str,
        feature_name: str,

) -> pd.DataFrame:
    """
    Liest alle .xmi-Dateien aus original_dir und simplified_dir ein,
    matcht sie über ihre ID, und baut ein DataFrame mit den Spalten:
        id, original_value, simplified_value

    Dieses DataFrame ist direkt kompatibel mit den Funktionen aus
    readability_stats.py (compute_basic_stats, compute_difference_stats).
    """

    # Schritt 1: Original-Werte einlesen, indiziert nach ID
    original_values = {}
    for filepath in sorted(Path(original_dir).glob("*.xmi")):
        file_id = extract_id_from_filename(filepath.name)
        value = extract_feature_value(filepath, typesystem, feature_type, feature_name)
        original_values[file_id] = value

    # Schritt 2: Simplified-Werte einlesen, indiziert nach ID
    simplified_values = {}
    for filepath in sorted(Path(simplified_dir).glob("*.xmi")):
        file_id = extract_id_from_filename(filepath.name)
        value = extract_feature_value(filepath, typesystem, feature_type, feature_name)
        simplified_values[file_id] = value

    # Schritt 3: Über gemeinsame IDs zusammenführen
    all_ids = sorted(set(original_values.keys()) | set(simplified_values.keys()))

    rows = []
    missing_pairs = []

    for file_id in all_ids:
        orig_val = original_values.get(file_id)
        simp_val = simplified_values.get(file_id)

        if orig_val is None or simp_val is None:
            missing_pairs.append(file_id)
            continue

        rows.append({
            "id": file_id,
            "original_value": orig_val,
            "simplified_value": simp_val,
        })

    if missing_pairs:
        print(f"Warnung: {len(missing_pairs)} IDs ohne vollständiges Paar (Original+Simplified):")
        for mid in missing_pairs[:10]:
            print(f"  {mid}")
        if len(missing_pairs) > 10:
            print(f"  ... und {len(missing_pairs) - 10} weitere")

    df = pd.DataFrame(rows)
    print(f"\n{len(df)} vollständige Paare erfolgreich eingelesen.")
    return df


"""
Statistik-Funktionen für Readability-Features über einen Datensatz.

Funktion 1: compute_basic_stats()
    Berechnet Durchschnitt, Maximum und Minimum eines Readability-Features
    über alle Werte einer Version (z.B. nur "Original" oder nur "Simplified").

Funktion 2: compute_difference_stats()
    Berechnet die Differenz (Original - Simplified) pro Satz-/Textpaar,
    und gibt davon Durchschnitt, größte Differenz und kleinste Differenz zurück.

Erwartetes Format: ein DataFrame mit mindestens den Spalten
    "original_value"   -> Readability-Score der Originalversion
    "simplified_value" -> Readability-Score der vereinfachten Version

Falls deine Spalten anders heißen, einfach beim Funktionsaufruf die
Spaltennamen als Parameter übergeben.
"""




def compute_basic_stats(df: pd.DataFrame, column: str) -> dict:
    """
    Berechnet Durchschnitt, Maximum und Minimum für eine einzelne Spalte
    (z.B. nur die Original- oder nur die Simplified-Werte).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mit den Readability-Werten.
    column : str
        Name der Spalte, für die die Statistik berechnet werden soll.

    Returns
    -------
    dict mit den Schlüsseln: "avg", "max", "min", "n"
    """
    values = df[column].dropna()

    stats = {
        "avg": values.mean(),
        "max": values.max(),
        "min": values.min(),
        "n": len(values),
    }
    return stats


"""
Statistik-Funktionen für Readability-Features über einen Datensatz.

Funktion 1: compute_basic_stats()
    Berechnet Durchschnitt, Maximum und Minimum eines Readability-Features
    über alle Werte einer Version (z.B. nur "Original" oder nur "Simplified").

Funktion 2: compute_difference_stats()
    Berechnet die Differenz (Original - Simplified) pro Satz-/Textpaar,
    und gibt davon Durchschnitt, größte Differenz und kleinste Differenz zurück.

Erwartetes Format: ein DataFrame mit mindestens den Spalten
    "original_value"   -> Readability-Score der Originalversion
    "simplified_value" -> Readability-Score der vereinfachten Version

Falls deine Spalten anders heißen, einfach beim Funktionsaufruf die
Spaltennamen als Parameter übergeben.
"""

import pandas as pd


def compute_basic_stats(df: pd.DataFrame, column: str) -> dict:
    """
    Berechnet Durchschnitt, Maximum und Minimum für eine einzelne Spalte
    (z.B. nur die Original- oder nur die Simplified-Werte).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mit den Readability-Werten.
    column : str
        Name der Spalte, für die die Statistik berechnet werden soll.

    Returns
    -------
    dict mit den Schlüsseln: "avg", "max", "min", "n"
    """
    values = df[column].dropna()

    stats = {
        "avg": values.mean(),
        "max": values.max(),
        "min": values.min(),
        "n": len(values),
    }
    return stats


def compute_difference_stats(
        df: pd.DataFrame,
        original_column: str = "original_value",
        simplified_column: str = "simplified_value",
) -> dict:
    """
    Berechnet die Differenz (Original - Simplified) für jedes Paar,
    und gibt davon Durchschnitt, größte und kleinste Differenz zurück.

    Eine positive Differenz bedeutet: Original hat einen höheren Wert
    als die vereinfachte Version (z.B. höherer Flesch-Score = leichter
    zu lesen würde dann eine NEGATIVE Differenz ergeben, wenn die
    vereinfachte Version leichter ist als das Original - je nach
    Interpretation deines Readability-Maßes beachten!).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mit den Readability-Werten für beide Versionen.
    original_column : str
        Spaltenname für die Original-Werte.
    simplified_column : str
        Spaltenname für die Simplified-Werte.

    Returns
    -------
    dict mit den Schlüsseln: "avg_diff", "max_diff", "min_diff", "n"
    sowie der vollständigen Differenz-Spalte als "diff_values" (pd.Series)
    """
    valid = df[[original_column, simplified_column]].dropna()

    diff = valid[original_column] - valid[simplified_column]

    stats = {
        "avg_diff": diff.mean(),
        "max_diff": diff.max(),  # größte Differenz (Original deutlich > Simplified)
        "min_diff": diff.min(),  # kleinste Differenz (kann auch negativ sein!)
        "n": len(diff),
        "diff_values": diff,
    }
    return stats


def get_feature_value(filepath, typesystem, feature_type=T_FEATURE, feature_name=FEATURE_NAME):
    with open(filepath, "rb") as f:
        cas = load_cas_from_xmi(f, typesystem=typesystem)
    for annotation in cas.select(feature_type):
        if annotation["name"] == feature_name:
            return float(annotation["value"])
    return None


def build_readability_dataframe(original_dir: str, simplified_dir: str) -> pd.DataFrame:

    rows = []
    for orig_path in sorted(Path(original_dir).glob("*.xmi")):
        file_id = orig_path.stem  # Dateiname ohne Endung = ID
        simp_path = Path(simplified_dir) / orig_path.name

        if not simp_path.exists():
            continue

        rows.append({
            "id": file_id,
            "original_value": get_feature_value(orig_path, typesystem),
            "simplified_value": get_feature_value(simp_path, typesystem),
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------
# Beispielnutzung
# ---------------------------------------------------------
if __name__ == "__main__":



    orig_dir = Path("datasets_lifted/deplain/text")
    o_out_dir = Path("datasets_lifted/renamed/deplain/text")
    o_out_dir.mkdir(parents=True, exist_ok=True)

    for xmi_file in orig_dir.glob("*.xmi"):
        new_name = keep_trailing_number_filename(xmi_file)
        target = o_out_dir / new_name
        shutil.copy2(xmi_file, target)

    simpl_dir = Path("datasets_lifted/deplain/simple")
    s_out_dir = Path("datasets_lifted/renamed/deplain/simple")
    s_out_dir.mkdir(parents=True, exist_ok=True)

    for xmi_file in simpl_dir.glob("*.xmi"):
        new_name = keep_trailing_number_filename(xmi_file)
        target = s_out_dir / new_name
        shutil.copy2(xmi_file, target)


    df = build_readability_dataframe(
        original_dir="datasets_lifted/renamed/deplain/text",
        simplified_dir="datasets_lifted/renamed/deplain/simple"
    )
    print('TEST')
    print(df.head())

    # Beispiel-Daten zum Testen
    example_df = pd.DataFrame({
        "id": ["1", "2", "3", "4"],
        "original_value": [45.2, 30.1, 60.5, 25.0],
        "simplified_value": [70.3, 55.0, 80.1, 40.2],
    })

    print("=== Statistik: Original-Werte ===")
    original_stats = compute_basic_stats(example_df, "original_value")
    print(original_stats)

    print("\n=== Statistik: Simplified-Werte ===")
    simplified_stats = compute_basic_stats(example_df, "simplified_value")
    print(simplified_stats)

    print("\n=== Statistik: Differenz (Original - Simplified) ===")
    diff_stats = compute_difference_stats(example_df)
    # diff_values separat ausgeben, damit die Konsole nicht überladen wird
    diff_values = diff_stats.pop("diff_values")
    print(diff_stats)
    print("\nEinzelne Differenzen pro Paar:")
    print(diff_values)