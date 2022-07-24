import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




df_dead_or_alive = pd.read_csv("./train.csv")

df_test = pd.read_csv("./test.csv")

mas_age_med = df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Master\.")]["Age"].median()
mr_age_med = df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Mr\.")]["Age"].median()
mis_age_med = df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Miss\.")]["Age"].median()
mrs_age_med = df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Mrs\.")]["Age"].median()
dr_age_med = df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Dr\.")]["Age"].median()


    
df_dead_or_alive.loc[df_dead_or_alive["Age"].isnull()&df_dead_or_alive["Name"].str.contains("Master\."), "Age"] = mas_age_med
df_dead_or_alive.loc[df_dead_or_alive["Age"].isnull()&df_dead_or_alive["Name"].str.contains("Mr\."), "Age"] = mr_age_med
df_dead_or_alive.loc[df_dead_or_alive["Age"].isnull()&df_dead_or_alive["Name"].str.contains("Miss\."), "Age"] = mis_age_med
df_dead_or_alive.loc[df_dead_or_alive["Age"].isnull()&df_dead_or_alive["Name"].str.contains("Mrs\."), "Age"] = mrs_age_med
df_dead_or_alive.loc[df_dead_or_alive["Age"].isnull()&df_dead_or_alive["Name"].str.contains("Dr\."), "Age"] = dr_age_med

df_dead_or_alive.loc[df_dead_or_alive["Sex"]=="male", "Sex"] = 0
df_dead_or_alive.loc[df_dead_or_alive["Sex"]=="female", "Sex"] = 1

df_test.loc[df_test["Age"].isnull()&df_test["Name"].str.contains("Master\."), "Age"] = mas_age_med
df_test.loc[df_test["Age"].isnull()&df_test["Name"].str.contains("Mr\."), "Age"] = mr_age_med
df_test.loc[df_test["Age"].isnull()&df_test["Name"].str.contains("Miss\."), "Age"] = mis_age_med
df_test.loc[df_test["Age"].isnull()&df_test["Name"].str.contains("Mrs\."), "Age"] = mrs_age_med
df_test.loc[df_test["Age"].isnull()&df_test["Name"].str.contains("Dr\."), "Age"] = dr_age_med

df_test.loc[df_test["Sex"]=="male", "Sex"] = 0
df_test.loc[df_test["Sex"]=="female", "Sex"] = 1



X = df_dead_or_alive.loc[:, ["Age", "Sex", "Pclass"]].to_numpy()
y = df_dead_or_alive["Survived"].to_numpy()

X_test = df_test.loc[:, ["Age", "Sex", "Pclass"]].to_numpy()

# tree = DecisionTreeClassifier(max_depth=4)
# tree.fit(X,y)
# print(tree.score(X,y))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

model = XGBClassifier(early_stopping_rounds=10, max_depth=4)
eval_set = [(X_val, y_val)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=True, early_stopping_rounds=10)

# export_graphviz(tree, out_file="tree.dot", class_names=["Dead", "Alive"],
#                 feature_names=["Age", "isFemale", "Pclass"], impurity=False, filled=True)

print(model.score)
model.feature_importances_
plt.barh(df_dead_or_alive.loc[:, ["Age", "Sex", "Pclass"]].columns, model.feature_importances_)
plt.show()


y_pred = model.predict_proba(X_test)
# print(len(np.argmax(y_pred, axis=1)))
ans = pd.DataFrame({"PassengerId": df_test["PassengerId"], "Survived":np.argmax(y_pred, axis=1)})

ans.to_csv("./ans.csv", index=False)

