import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics._classification import classification_report


data = pd.read_csv('Development Index.csv')
data = np.array(data)

stat, p = shapiro(data[0])
print("stat = ", stat, "pvalue = ", p)
shapiro_test = stats.shapiro(data)

df = pd.DataFrame({
'population':data[:,0],
'density':data[:,2],
'gdp':data[:,3],
'devindex':data[:,6]
})
df.head()
print(df)

x = df[['population', 'density', 'gdp']].values
y = df['devindex'].values
print("y: ", y)

"""
x = (x-x.min())/(x.max()-x.min())
y = (y-y.min())/(y.max()-y.min())
print("x: ", x)
print("y: ", y)
"""



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

x_trainDN = x_train
x_testDN = x_test
y_trainDN = y_train
y_testDN = y_test


scaler = MinMaxScaler(feature_range=(0, 1))
xScale = scaler.fit(x)
x_train = xScale.transform(x_train)
x_test = xScale.transform(x_test)

yScale = scaler.fit(y_train.reshape(len(y_train), 1))
y_train = yScale.transform(y_train.reshape(len(y_train), 1))
#y_test = yScale.transform(y_test.reshape(len(y_train), 1))

"""
x_train=(x_train-x_train.min())/(x_train.max()-x_train.min())
x_test=(x_test-x_test.min())/(x_test.max()-x_test.min())
y_train=(y_train-y_train.min())/(y_train.max()-y_train.min())
y_test=(y_test-y_test.min())/(y_test.max()-y_test.min())

print("x_train: ", x_train)
print("y_train: ", y_train)
#print(x_train)
"""
model = Sequential(
    [
        Dense(7, activation='relu', name="layer1"),
        Dense(7, activation='relu', name="layer2"),
        Dense(7, activation='relu', name="layer3"),
        #Dense(9, activation='relu', name="layer4"),
        Dense(1, activation='sigmoid', name="layerExit"),
    ])

adagrad = tf.keras.optimizers.Adagrad(learning_rate=0.1, name='Adagrad')
model.compile(optimizer=adagrad, loss=tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
res = model.fit(x_train, y_train, epochs=30, batch_size=13, validation_data=(x_test, y_test))

print(res)
result = model.evaluate(x_test, y_test, batch_size=13)
print('test loss, test acc:', result)

pre = model.predict(x_test)
#pre *= pre.max()
#print(pre)
#pre = y.inverse_transform(pre)
#pre = np.rint(pre)

pre = yScale.inverse_transform(pre)
pre = np.rint(pre)
#print("pre: ", pre)
print(confusion_matrix(y_testDN, pre))
print("report", classification_report(y_testDN, pre))


data = pd.DataFrame({'real': y_testDN.flatten(), 'predicted': pre.flatten().round()})
crop = data.head(30)
crop.plot(kind="bar")
plt.grid()
plt.show()


#plt.hist(pre,  alpha = 0.5, color="blue")
#plt.hist(y_test, alpha = 0.5, color="orange")

#rfc = RandomForestClassifier(n_estimators=40, random_state=3)
#res = cross_val_score(rfc, x, y, cv=5)
#mean = np.mean(res)
#print("Среднее кросс-валидации: ", mean)



#print("res: ", res)

#rfc.fit(x_train, y_train)
"""
def plot_conf_matrix(cm, classes, normalize=False, title="Матрица ошибок", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print('Матрица ошибок')
    print(cm)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('Истинные значения')
    plt.xlabel('Предсказанные значения')
"""
"""
y_prediction = rfc.predict(x_test)

f1 = f1_score(y_test.values, y_prediction, average="micro")
print("f1score: ", f1)

print("Prediction is: ", y_prediction)
print("Real is:", y_test.values)

font = {'size' : 15}
plt.rc('font', **font)

cnf_matrix = confusion_matrix(y_test, y_prediction)
print("y_test: \n", y_test)
print("y_pre: \n",  y_prediction)
print("matrix: \n", cnf_matrix)
plt.figure(figsize=(10, 8))
plot_conf_matrix(cnf_matrix,
                classes=['x[0], x[2], x[3], y'],
                 title='Матрица ошибок')

#plt.savefig('conf_matrix.png')
plt.show()
"""