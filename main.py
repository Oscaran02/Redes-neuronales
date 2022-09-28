import pandas as pd

# Import the data into a pandas dataframe
path = 'Data/'
dataframe = []
for t in range(15, 46):
    for g in range(400, 1000, 20):
        filename = f'OutputPV_T{t}_G{g}.txt'
        df = pd.read_csv(path + filename, header=None)
        header = ['V', 'I']
        df.columns = header
        df['P'] = df['V'] * df['I']
        df['T'] = ""
        df['G'] = ""
        df = df.assign(T=t, G=g)
        dataframe.append(df)
DF = pd.concat(dataframe, ignore_index=True)
print(DF)
