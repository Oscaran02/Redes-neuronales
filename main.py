import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Import the data into a pandas dataframe
path = 'Data/'
dataframe = []
for t in range(15, 46):
    for g in range(400, 1001, 20):
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

train, test = train_test_split(DF, test_size=0.3)

g = train['G']
t = train['T']
v = train['V']

entry = np.array(np.column_stack((g, t, v)), dtype=np.float)

i = train['I']
out = np.array(np.transpose(i), dtype=np.float)

# Hidden layers
layer1 = tf.keras.layers.Dense(units=2, input_shape=[3])
model = tf.keras.Sequential([layer1])

# Compile the model
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# Train the model
print("Training the model...")
history = model.fit(entry, out, epochs=100, verbose=False)
print("Finished training the model")