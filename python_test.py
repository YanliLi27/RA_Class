import pandas as pd


df: pd.DataFrame = pd.DataFrame([
    [1, 1, 1, 1, 1],
    [2, 2, None,2,2],
    [3,3,3,3,3]
], columns=['one', 'two', 'three', 'four', 'five'])

titles = [column for column in df]
ramris_ids = df[titles[0]].values
print(ramris_ids)
print(type(ramris_ids))


print(df.iloc[1, 1:2].values)
print(type(df.iloc[1, 1:2].values))
