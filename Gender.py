from sklearn import tree
from sklearn import svm
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

clf1 = tree.DecisionTreeClassifier()
clf2 = svm.SVC()
clf3 = QuadraticDiscriminantAnalysis()
clf4 = AdaBoostClassifier()
clf5 = RandomForestClassifier()
clf6 = GaussianProcessClassifier(1.0 * RBF(1.0))
clf7 = GaussianNB()
clf8 = KNeighborsClassifier()
clf9 = MLPClassifier(max_iter=1000)


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)
clf4 = clf4.fit(X, Y)
clf5 = clf5.fit(X, Y)
clf6 = clf6.fit(X, Y)
clf7 = clf7.fit(X, Y)
clf8 = clf8.fit(X, Y)
clf9 = clf9.fit(X, Y)

prediction1 = clf1.predict([[190, 70, 43]])
prediction2 = clf2.predict([[190, 70, 43]])
prediction3 = clf3.predict([[190, 70, 43]]) 
prediction4 = clf4.predict([[190, 70, 43]])
prediction5 = clf5.predict([[190, 70, 43]])
prediction6 = clf6.predict([[190, 70, 43]])
prediction7 = clf7.predict([[190, 70, 43]])
prediction8 = clf8.predict([[190, 70, 43]])
prediction9 = clf9.predict([[190, 70, 43]])

print(prediction1) #male
print(prediction2) #male
print(prediction3) #male
print(prediction4) #male
print(prediction5) #male
print(prediction6) #male
print(prediction7) #male
print(prediction8) #male
print(prediction9) #female