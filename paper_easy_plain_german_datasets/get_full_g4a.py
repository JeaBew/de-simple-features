from pathlib import Path
from datasets import load_dataset, concatenate_datasets

german4all_corrected = load_dataset("tum-nlp/German4All-Corpus", data_dir="corrected")

# Splits holen (oder direkt: ["train"], ["test"], je nach Dataset)
split_train = list(german4all_corrected.keys())[0]
split_test  = list(german4all_corrected.keys())[1]
split_three = list(german4all_corrected.keys())[2]

ds_train = german4all_corrected[split_train]
ds_test  = german4all_corrected[split_test]
ds_three = german4all_corrected[split_three]

# Zusammenführen zu einem Dataset
ds_all = concatenate_datasets([ds_train, ds_test, ds_three])

out_csv = Path("datasets_raw/german4all_corrected/german4all_corrected_full.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)

ds_all.to_csv(str(out_csv))
print(f"Gespeichert als CSV unter: {out_csv}")
print("Gesamtanzahl Zeilen:", len(ds_all))