import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

df_dead_or_alive = pd.read_csv("./train.csv")

#print(df_dead_or_alive["Survived"])

#sns.scatterplot3d(x="Age", y='Fare', z="Sex", hue=["Survived"], data=df_dead_or_alive)
#pg = sns.pairplot(df_dead_or_alive, hue="Sex")
#plt.show()


sns.set_style("darkgrid")

mean = 3
number =50

x1 = df_dead_or_alive["Age"]
y1 = df_dead_or_alive["Pclass"]
z1 = df_dead_or_alive["Sex"]=="male"
col = ["blue","orange"]
lbl = [col[t] for t in df_dead_or_alive["Survived"]]

plt.figure(figsize=(18,18))
axes = plt.axes(projection='3d')
axes.scatter3D(x1, y1, z1, c=lbl)

axes.set_xlabel('Age')
axes.set_ylabel('Fare')
axes.set_zlabel('Sex_is_male')

#plt.show()
female_alive =  len(df_dead_or_alive.query("Sex=='female'and Survived==1 and Age<=10")) /len(df_dead_or_alive.query("Sex=='female' and Age<=10"))
print(female_alive)

male_alive =  len(df_dead_or_alive.query("Sex=='female'and Survived==1")) /(df_dead_or_alive["Sex"]=="male").sum()
print(male_alive)

print(df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Master\.")]["Age"].describe())
print(df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Mr\.")]["Age"].describe())
print(df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Miss\.")]["Age"].describe())
print(df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Ms\.")]["Age"].describe())
print(df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Mrs\.")]["Age"].describe())
print(df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Don\.")]["Age"].describe())
print(df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Master\.")]["Name"].describe())
print(df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Mr\.")]["Name"].describe())
print(df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Miss\.")]["Name"].describe())
print(df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Ms\.")]["Name"].describe())
print(df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Mrs\.")]["Name"].describe())
print(df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Don\.|Dona\.")]["Name"].describe())
print(df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Dr\.")]["Name"].describe())
print(df_dead_or_alive[df_dead_or_alive["Name"].str.contains("Capt\.")]["Name"].describe())
print(df_dead_or_alive["Name"].describe())