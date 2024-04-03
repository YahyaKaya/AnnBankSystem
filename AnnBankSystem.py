import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv(r"address of the cvs file")
dataset.head()

x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

x[:,2] = le.fit_transform(x[:,2])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[1])], remainder='passthrough')

x = np.array(ct.fit_transform(x))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state= 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(x_train, y_train, batch_size= 32, epochs=100)

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 1, 80000, 1, 0, 1, 1100000]])))
print("Will stay?", ann.predict(sc.transform([[1, 0, 0, 500, 2, 40, 8, 160000, 2, 1, 1, 110000]])) > 0.05)


y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)