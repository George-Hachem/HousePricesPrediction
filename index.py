import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy import stats








##python3 -u "/Users/george/HousePricesPredictions/HousePricesPrediction/index.py"
housing = pd.read_csv('housing.csv')

##Data info
# print(housing.head())
# print(housing.info())
#
##Count ocean_proximity districts
#count_districts = housing["ocean_proximity"].value_counts()
#print(count_districts)

##Check other numerical attributes
#print(housing.describe())

#Draw Histogram
# housing.hist(bins=50, figsize=(20,15))
# plt.show()

##CReate a test data
# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]

#train_set, test_set = split_train_test(housing, 0.2)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# print(len(train_set))
# print(len(test_set))

## Graph median income
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

#housing["income_cat"].hist()
#plt.show()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


housing = strat_train_set.copy()

# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
#plt.legend()
# plt.show()

## Correlation between attributes

# corr_matrix = housing.corr()

# print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
# plt.show()


### Prepare data for machine learning

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

##Data Cleaning
#Fill the missing total_bedroom values with median values
median = housing["total_bedrooms"].median()  # option 3
# housing["total_bedrooms"].fillna(median, inplace=True)

imputer = SimpleImputer(strategy="median") #Fill all missing values of eaxh attribute with their median

housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

# print(imputer.statistics_)
# print(housing_num.median().values)

#Tranform the data and modify the missing values
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

###Handling texts and categorical attributes

##Convert category from text to numbers
housing_cat = housing[['ocean_proximity']]

# ordinal_encoder = OrdinalEncoder()
# housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# print(housing_cat_encoded[:10])
#print(ordinal_encoder.categories_)


##One hot encoding
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# print(housing_cat_1hot.toarray())

# print(cat_encoder.categories_)

##Custom Transformers
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]

        else:
            return np.c_[X, rooms_per_household, population_per_household]

# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs = attr_adder.transform(housing.values)

##Transformation Pipelines
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
])


##Apply appropraite tranformations to each column
housing_num_tr = num_pipeline.fit_transform(housing_num)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)


###Select and Train a model

##Use linear regression model

# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)
# some_data = housing.iloc[:5]
# some_labels = housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)
# print("Predictions:", lin_reg.predict(some_data_prepared))
# print("Labels:", list(some_labels))

##Mean squared error
# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
#print(lin_rmse) #The mean squared error is $68627, which is not good, we are underfitting the data. Lets try a new model

##Use DecisionTreeRegressor (worls best with nonlinear data)

# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)

# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_rmse = np.sqrt(tree_mse)
#print(tree_rmse) # we get 0.0, which means it is showing no error, but the model most probably overfit the data


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
#                          scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-scores)

# lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
#                             scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores = np.sqrt(-lin_scores)


#print(display_scores(tree_rmse_scores)) #It seems like DecisionTreeRegressor prefroms worse than linear regression
#print(display_scores(lin_rmse_scores)) # Linear Regression prefroms better than DecisionTreeRegressor


##Use RandomForestRegressor model
# forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared, housing_labels)
# housing_predictions = forest_reg.predict(housing_prepared)
# forest_mse = mean_squared_error(housing_labels, housing_predictions)
# forest_rmse = np.sqrt(forest_mse)
# #print(forest_rmse)

# forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
#                                 scoring = "neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
#print(display_scores(forest_rmse_scores)) #RandomRorestRegressor preformed the best

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
# print(grid_search.best_params_)
# print(grid_search.best_estimator_)

# cvres = grid_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params) # by setting max_features hyperparameter to 6 and n_estimators to 30, the RMSE score for this combination is 49954 which is better than before
#The model is now fine-tuned


##Fine-tune the model even further
feature_importances = grid_search.best_estimator_.feature_importances_
#print(feature_importances)

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
#print(sorted(zip(feature_importances, attributes), reverse=True))

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

#print(final_rmse) # The RMSE value is now 47723


## I need to know how precise the model is, so Im going to use 95% confidence interval

confidence = 0.95

squared_errors = (final_predictions - y_test) ** 2
print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                          loc=squared_errors.mean(),
                          scale=stats.sem(squared_errors))))
#output: [45605.64479292 49390.47690802]

"""
the final performance of the system is not better than the experts’ price estimates,
which were often off by about 20%, but it may still be 
a good idea to launch it, especially if this frees up some time 
or the experts so they can work on more interesting and productive tasks
"""

