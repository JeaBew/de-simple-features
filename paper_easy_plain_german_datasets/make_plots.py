from cassis import load_cas_from_xmi
from py_lift.utils.core import load_lift_typesystem
from py_lift.dkpro import T_FEATURE
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

typesystem = load_lift_typesystem()

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

plot_deplain()