import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine

wine = load_wine()

from sklearn.model_selection import train_test_split, cross_val_score

Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, random_state=0,test_size=0.3)
clf = DecisionTreeClassifier(random_state=0)
rcf = RandomForestClassifier(random_state=0)
clf=clf.fit(Xtrain,Ytrain)
rcf=rcf.fit(Xtrain,Ytrain)

score_c=clf.score(Xtest,Ytest)
score_r=rcf.score(Xtest,Ytest)
print(f"single tree {score_c}")
print(f"random forest {score_r}")
superpa = []
for i in range(200):
    print(i)
    rfc = RandomForestClassifier(n_estimators=i+1,n_jobs=-1)
    rfc_s = cross_val_score(rfc,wine.data,wine.target,cv = 10).mean()
    superpa.append(rfc_s)
print(max(superpa),superpa.index(max(superpa)))
plt.figure(figsize=[20,5])
plt.show(range(1,201),superpa)
plt.show()