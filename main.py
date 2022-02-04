import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import extraction_functions as efs
import RNNLstm as lstm

from user_interface import recordedAudio as ra

warnings.filterwarnings('ignore')



Tess="C:\\Users\\Dell\\Desktop\\major project\\SERsystem\\TESS\\"

#loading the dataset in vs code
paths = []
labels = []
for dirname, _, filenames in os.walk(Tess):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        if label=='ps':
            labels.append('surprise')
        else:
            labels.append(label.lower())
    if len(paths) == 2800:
        break
# dataframe for emotion of files
emotion_df = pd.DataFrame(labels, columns=['Emotions'])
# dataframe for path of files.
path_df = pd.DataFrame(paths, columns=['Path'])
df = pd.concat([emotion_df, path_df], axis=1)
df.Emotions.replace({'neutral':0, 'surprise':1, 'happy':2, 'sad':3, 'angry':4, 'fear':5,'disgust':6}, inplace=True)
df.to_csv("feture_paths.csv",index=False)
df.head()


#printing the counts of each classes present
print(df['Emotions'].value_counts())


#
#
#iterating over the dataset and getting the features in array/table
Z=np.empty((0,90))
Y=[]
for path, emotion in zip(df.Path, df.Emotions):
  feature = efs.get_features(path)
  #print(feature)
  X= np.array([])
  for ftr in feature:
    X=np.hstack((X,ftr))
  Y.append(emotion)
  Z=np.vstack((Z,X))
#print(Z)


#
#
#the extracted data is shown in a table
Features = pd.DataFrame(Z)
Features['category'] = Y
Features.to_csv('features.csv', index=False)
Features.head()


#
#load X an Y from the dataframe
X = Features.iloc[: ,:-1].values
Y = Features['category'].values


#feautre selection for effiecient model
#stop the search when only the last 40 feature are left
lr = LinearRegression(normalize=True)
lr.fit(X,Y)
rfe = RFE(estimator = lr, n_features_to_select=60, step=1)
X = rfe.fit_transform(X,Y)


# splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

#building a model
input_shape = (x_train.shape[1],1)
model = lstm.build_model(input_shape)

#
#
history=model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_test, y_test))


#printing the accuracy of the model and the loss and accuracy graph
print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")
epochs = [i for i in range(30)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']

fig.set_size_inches(20,6)
ax[0].plot(epochs , train_loss , label = 'Training Loss')
ax[0].plot(epochs , test_loss , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()



# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

# predicting on test data.
pred_test = model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)




# #creating the table for the prediction on test data
Predicted_df = pd.DataFrame(columns=['PredictedLabels', 'ActualLabels'])
Predicted_df['PredictedLabels'] = y_pred.flatten()
Predicted_df['ActualLabels'] = y_test.flatten()
Predicted_df.PredictedLabels.replace({0:'neutral', 1:'surprise', 2:'happy', 3:'sad', 4:'angry', 5:'fear', 6:'disgust'}, inplace=True)
Predicted_df.ActualLabels.replace({0:'neutral', 1:'surprise', 2:'happy', 3:'sad', 4:'angry', 5:'fear', 6:'disgust'}, inplace=True)

print(Predicted_df.head(20))

print(classification_report(y_test, y_pred))

