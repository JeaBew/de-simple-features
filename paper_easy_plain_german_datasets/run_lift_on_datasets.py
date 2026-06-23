import pandas as pd
from cassis import cas_to_comparable_text
from py_lift.preprocessing import Spacy_Preprocessor
from py_lift.utils.core import load_lift_typesystem
from py_lift.readability import FE_TextstatFleschIndex
from py_lift.annotators.misc import SE_CoarsePosTagAnnotator
from pathlib import Path

prep = Spacy_Preprocessor("de", auto_install_models=True)
ts = load_lift_typesystem()


def lift_deplain():

    df = pd.read_csv("datasets_raw/DEPlain/deplain.csv")

    # text standard, cl_LS easy
    print(df.columns.tolist())
    subset = df[["original", "simplification", "pair_id"]]

    print(subset.head())

    output_dir_text = Path("datasets_lifted/deplain/text")
    output_dir_simple = Path("datasets_lifted/deplain/simple")
    for index, row in subset.iterrows():
        text = row["original"]
        simple = row["simplification"]
        docid = str(row["pair_id"])
        #print(f"Zeile {index}: {text[:50]}... -> {label}")

        cas_text = prep.run(text)
        cas_simple = prep.run(simple)

        FE_TextstatFleschIndex("de").extract(cas_text)
        FE_TextstatFleschIndex("de").extract(cas_simple)

        outfile_text = output_dir_text / f"deplain_text_{docid}.xmi"
        outfile_text.write_text(cas_text.to_xmi(), encoding="utf-8")

        outfile_simple = output_dir_simple / f"deplain_simple_{docid}.xmi"
        outfile_simple.write_text(cas_simple.to_xmi(), encoding="utf-8")



def lift_g4a():

    df = pd.read_csv("datasets_raw/german4all_corrected/german4all_corrected.csv")

    # text standard, cl_LS easy
    print(df.columns.tolist())
    subset = df[["text", "cl_LS", "id"]]

    print(subset.head())

    output_dir_text = Path("datasets_lifted/g4a/text")
    output_dir_simple = Path("datasets_lifted/g4a/simple")
    for index, row in subset.iterrows():
        text = row["text"]
        simple = row["cl_LS"]
        docid = str(row["id"])
        # print(f"Zeile {index}: {text[:50]}... -> {label}")

        cas_text = prep.run(text)
        cas_simple = prep.run(simple)

        FE_TextstatFleschIndex("de").extract(cas_text)
        FE_TextstatFleschIndex("de").extract(cas_simple)

        outfile_text = output_dir_text / f"g4a_text_{docid}.xmi"
        outfile_text.write_text(cas_text.to_xmi(), encoding="utf-8")

        outfile_simple = output_dir_simple / f"g4a_simple_{docid}.xmi"
        outfile_simple.write_text(cas_simple.to_xmi(), encoding="utf-8")

def lift_asgc():

    df = pd.read_csv("datasets_raw/ASGC/to_csv/asgc.csv")

    # text standard, cl_LS easy
    print(df.columns.tolist())
    subset = df[["original", "simple", "id"]]

    print(subset.head())

    output_dir_text = Path("datasets_lifted/asgc/text")
    output_dir_simple = Path("datasets_lifted/asgc/simple")
    for index, row in subset.iterrows():
        text = row["original"]
        simple = row["simple"]
        docid = str(row["id"])
        # print(f"Zeile {index}: {text[:50]}... -> {label}")

        cas_text = prep.run(text)
        cas_simple = prep.run(simple)

        FE_TextstatFleschIndex("de").extract(cas_text)
        FE_TextstatFleschIndex("de").extract(cas_simple)

        outfile_text = output_dir_text / f"asgc_text_{docid}.xmi"
        outfile_text.write_text(cas_text.to_xmi(), encoding="utf-8")

        outfile_simple = output_dir_simple / f"asgc_simple_{docid}.xmi"
        outfile_simple.write_text(cas_simple.to_xmi(), encoding="utf-8")


lift_deplain()
lift_g4a()
lift_asgc()