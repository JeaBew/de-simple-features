from datasets import load_dataset

# column name: cl_LS
# Load the different datasets with the `data_dir` parameter
#german4all_main = load_dataset("tum-nlp/German4All-Corpus", data_dir="main")
german4all_corrected = load_dataset("tum-nlp/German4All-Corpus", data_dir="corrected")

print(german4all_corrected)

split_name = list(german4all_corrected.keys())[0]
data = german4all_corrected[split_name]

print(f"\nSplit: {split_name}")
print(f"Anzahl Zeilen: {len(data)}")
print(f"\nSpaltennamen:\n{data.column_names}")

print("\nBeispielzeile (erste Zeile, gekürzt):")
example = data[0]
for key, value in example.items():
    value_str = str(value)
    if len(value_str) > 200:
        value_str = value_str[:200] + " ..."
    print(f"  {key}: {value_str}")

out_csv = "datasets_raw/german4all_corrected/german4all_corrected.csv"
data.to_csv(out_csv)
print(f"Gespeichert als CSV unter: {out_csv}")