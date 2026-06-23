import pandas as pd
import py_lift



def lift_deplain():

    df = pd.read_csv("datasets_raw/DEPlain/deplain.csv")

    # text standard, cl_LS easy
    print(df.columns.tolist())
    subset = df[["original", "simplification"]]

    print(subset.head())

    



def lift_g4a():

    df = pd.read_csv("datasets_raw/german4all_corrected/german4all_corrected.csv")

    # text standard, cl_LS easy
    print(df.columns.tolist())
    subset = df[["text", "cl_LS"]]

    print(subset.head())

def lift_asgc():

    df = pd.read_csv("datasets_raw/ASGC/to_csv/asgc.csv")

    # text standard, cl_LS easy
    print(df.columns.tolist())
    subset = df[["original", "simple"]]

    print(subset.head())


lift_deplain()
lift_g4a()
lift_asgc()