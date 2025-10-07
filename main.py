import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. load the dataset
housing = pd.read_csv("housing.csv")

# 2. create stratified test set
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1,2,3,4,5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1) #set aside test data

#work on copy of training data
housing = strat_train_set.copy()

# 3. separate features and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop ("median_house_value", axis=1)

# 4. separate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# 5. create pipeline 

# for numerical columns
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler())
])

# for categorical columns
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown='ignore'))
])

# final full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# 6. transform the data
housing_prepared = full_pipeline.fit_transform(housing)

# 7. training model

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds= lin_reg.predict(housing_prepared)
# scikit learn scoring uses utility functions (higher is better), so rmse is retured as negative, so we need to negate it to get positive value
lin_rmse = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10) # cv = cross validation 
print("this is linear regression:")
print(pd.Series(lin_rmse).describe())
# lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
# print(f"the root mean squared error for linear regression is {lin_rmse}")


# decision tree
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_preds= dec_reg.predict(housing_prepared)
# dec_rmse = root_mean_squared_error(housing_labels, dec_preds) #this will overfit the model here
dec_rmse = -cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10) # cv = cross validation 
print("this is decision tree:")
# for calculating the avg of rmse
print(pd.Series(dec_rmse).describe())

# print(f"the root mean squared error for decision tree is {dec_rmse}")

# random forest 
random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(housing_prepared, housing_labels)
random_forest_preds= random_forest_reg.predict(housing_prepared)
random_forest_rmse = -cross_val_score(random_forest_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10) # cv = cross validation 
print("this is random forest:")
print(pd.Series(random_forest_rmse).describe())
# random_forest_rmse = root_mean_squared_error(housing_labels, random_forest_preds)
# print(f"the root mean squared error for random forest is {random_forest_rmse}")


# random forest gives the lowest rmse, so using this model for test data
# training rmse will only show how well model fits training data, it does not tell how well it will perform on unseen data
# here, decision tree and random forest may overfit, leading to very low training error but poor generalization
# so, used cross-validation here