import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. Load the Data 
data = pd.read_csv('housing.csv')

# 2. Create a stratified test set based on income category
data['income_cat'] = pd.cut(data['median_income'], 
                            bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], 
                            labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data, data['income_cat']):
    set_train_test = data.loc[train_index].drop('income_cat', axis=1)
    set_test_test = data.loc[test_index].drop('income_cat', axis=1)


# 3. Work on a copy of training data 
data = set_train_test.copy() 

# 4. Seprate Predictors and labels  
housing_labels = data['median_house_value'].copy()
data = data.drop('median_house_value', axis=1)

# 5. Separate numerical and categorical columns
num_attribs = data.drop("ocean_proximity", axis=1).columns.tolist() 
cat_attribs = ["ocean_proximity"]


print(num_attribs)
print(cat_attribs)

# 6. Numerical Pipelines 
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler()),
])

# 7. Categorical Pipeline
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 8. Full Pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs), 
    ("cat", cat_pipeline, cat_attribs),
]) 

# 9. Transform the data 

housing_prepared = full_pipeline.fit_transform(data)

# print(housing_prepared) 

