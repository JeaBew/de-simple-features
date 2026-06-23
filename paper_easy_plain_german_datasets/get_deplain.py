from datasets import load_dataset

deplain = load_dataset("DEplain/DEplain-web-doc")

print(deplain)

split_name = list(deplain.keys())[0]
data = deplain[split_name]

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

out_csv = "datasets_raw/deplain/deplain.csv"
data.to_csv(out_csv)
print(f"Gespeichert als CSV unter: {out_csv}")