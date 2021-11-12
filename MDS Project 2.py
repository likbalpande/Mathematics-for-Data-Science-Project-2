from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# reading data
col_names = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height",
             "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
df = pd.read_csv("./data_set.data", names=col_names, na_values=['?'])

""" 
df.info(verbose=True)
output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 204 entries, 0 to 203
Data columns (total 26 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   symboling          204 non-null    int64  
 1   normalized-losses  163 non-null    float64
 2   make               204 non-null    object 
 3   fuel-type          204 non-null    object 
 4   aspiration         204 non-null    object 
 5   num-of-doors       202 non-null    object 
 6   body-style         204 non-null    object 
 7   drive-wheels       204 non-null    object 
 8   engine-location    204 non-null    object 
 9   wheel-base         204 non-null    float64
 10  length             204 non-null    float64
 11  width              204 non-null    float64
 12  height             204 non-null    float64
 13  curb-weight        204 non-null    int64  
 14  engine-type        204 non-null    object 
 15  num-of-cylinders   204 non-null    object 
 16  engine-size        204 non-null    int64  
 17  fuel-system        204 non-null    object 
 18  bore               200 non-null    float64
 19  stroke             200 non-null    float64
 20  compression-ratio  204 non-null    float64
 21  horsepower         202 non-null    float64
 22  peak-rpm           202 non-null    float64
 23  city-mpg           204 non-null    int64  
 24  highway-mpg        204 non-null    int64  
 25  price              200 non-null    float64
dtypes: float64(11), int64(5), object(10)
memory usage: 41.6+ KB 


# get dummies for every column with type object
d_make = pd.get_dummies(df['make'])
d_fuel_type = pd.get_dummies(df['fuel-type'])
d_aspiration = pd.get_dummies(df['aspiration'])
d_num_of_doors = pd.get_dummies(df['num-of-doors'], prefix="d")
d_body_style = pd.get_dummies(df['body-style'])
d_drive_wheels = pd.get_dummies(df['drive-wheels'])
d_engine_location = pd.get_dummies(df['engine-location'])
d_engine_type = pd.get_dummies(df['engine-type'])
d_num_of_cylinders = pd.get_dummies(df['num-of-cylinders'], prefix="noc")
d_fuel_system = pd.get_dummies(df['fuel-system'])

df = df.join(d_make)
df = df.join(d_fuel_type)
df = df.join(d_aspiration)
df = df.join(d_num_of_doors)
df = df.join(d_body_style)
df = df.join(d_drive_wheels)
df = df.join(d_engine_location)
df = df.join(d_engine_type)
df = df.join(d_num_of_cylinders)
df = df.join(d_fuel_system)

object_cols = [
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "engine-type",
    "num-of-cylinders",
    "fuel-system"
]
df = df.drop(object_cols, axis=1)
df.fillna(df.mean(), inplace=True)
df.to_csv("df.csv")
predictions = [
    "symboling",
]
""" features = df.columns - predictions """

X = df.drop(predictions, axis=1)
y = df[predictions]
X_train, X_test, y_train, y_test = train_test_split(
    X, y.values.ravel(), test_size=0.2, random_state=42)
SS = StandardScaler()
X_train = SS.fit_transform(X_train)
X_test = SS.transform(X_test)
clf = KNeighborsClassifier(n_neighbors=10)
clf1 = LogisticRegression(random_state=0,
                         solver="saga", max_iter=20000)
clf2 = LogisticRegression(penalty="l1",random_state=0,
                         solver="saga", max_iter=20000)
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
data = [106.0,97.2,173.4,65.2,54.7,2302,120,3.33,3.47,8.5,97.0,5200.0,27,34,9549.0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0]
y_pred1 = clf1.predict(np.array(data).reshape(1,-1))
print(y_pred1)
y_pred2 = clf2.predict(np.array(data).reshape(1,-1))
print(y_pred2)

print(clf1.score(X_test, y_test))
print(clf2.score(X_test, y_test))
