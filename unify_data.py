import os
import pandas as pd

json_filenames = [os.path.join('data', f)
                  for f in os.listdir('data') if 'non-troll' in f]

dataframes = []
for json_filename in json_filenames:
    print("Reading {}...".format(json_filename))
    file_df = pd.read_json(json_filename)
    dataframes.append(file_df)
print("Aggregating dataframes...")
aggregated_df = pd.concat(dataframes)
target_filename = os.path.join("data", "non_troll_data.csv")
print("Saving aggregated df to {}".format(target_filename))
aggregated_df.to_csv(target_filename)
print("Done.")
