import  numpy as np
import pandas as pd
import glob,string
path = 'datasets/coil-100/coil-100/*.png'
#list files
files=glob.glob(path)

from tqdm import tqdm
def contructDataframe(file_list):
    """
    this function builds a data frame which contains
    the path to image and the tag/object name using the prefix of the image name
    """
    data=[]
    for file in tqdm(file_list):
        data.append((file,file.split("/")[-1].split("__")[0]))
    return pd.DataFrame(data,columns=['path','label'])
df=contructDataframe(files)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.path, df.label, test_size=0.15,random_state=0,stratify= df.label)

X_train.groupby(y_train).size().reset_index(name="counts")
X_test.groupby(y_test).size().reset_index(name="counts")

from keras.preprocessing.image import img_to_array
import cv2
X_train=[img_to_array(cv2.imread(file).astype("float")/255.0) for file in tqdm(X_train.values)]
X_test=[img_to_array(cv2.imread(file).astype("float")/255.0) for file in tqdm(X_test.values)]
x = np.concatenate((X_train, X_test))
x = x.reshape((x.shape[0], -1))
np.savetxt('datasets/coil100.csv', x, delimiter=',', fmt='%f')

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
y_train_categorical=encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test_categorical=encoder.transform(y_test.values.reshape(-1, 1))

y = np.concatenate((y_train_categorical, y_test_categorical))
np.savetxt('datasets/coil100_label.csv', y, delimiter=',', fmt='%f')
