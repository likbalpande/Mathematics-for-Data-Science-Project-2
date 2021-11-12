import pandas as pd
import numpy as np
from pandas.core.common import random_state
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df = pd.read_csv(r"data_set.data",names = ['sym','loss','make','fuel_t','asp','n_doors','body','wheels','engine_l','wheel_b','length','width','height','c_height','engine_t','n_cylinders','engine_s','fuel_s','bore','stroke','c_ratio','horsepower','p_rpm','c_mpg','h_mpg','price'], na_values = ['?'])
df.loc[len(df.index)] = [1,106,"nissan","gas","std","four","sedan","fwd","front",97.20,173.40,65.20,54.70,2302,"ohc","four",120,"2bbl",3.33,3.47,8.50,97,5200,27,34,9549]
features = df.drop('sym', 1)
pred = df['sym']
numeric_cols = ['loss','stroke','horsepower','p_rpm','price','bore']
string_cols = ['make','fuel_t','asp','n_doors','body','wheels','engine_l','engine_t','n_cylinders','fuel_s']
#dummies_cols = ['d_make','d_fuel_t','d_asp','d_n_doors','d_body','d_wheels','d_engine_l','d_engine_t','d_n_cylinders','d_fuel_s']
for col in numeric_cols:
    features[col] = features[col].fillna(features[col].mean())
n_mode = features['n_doors'].mode()
features[['n_doors']] = features[['n_doors']].replace(np.nan,n_mode[0])
for col in string_cols:
    if col == 'n_doors' :
        features = features.join(pd.get_dummies(features[col],prefix='nd'))
    elif col == 'n_cylinders' :
        features = features.join(pd.get_dummies(features[col],prefix='nc'))
    else:
        features = features.join(pd.get_dummies(features[col]))
features = features.drop(string_cols,1)
features.to_csv('df.csv')
