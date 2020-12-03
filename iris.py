from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score  # K折交叉验证模块
from sklearn.externals import joblib  # jbolib模块
import matplotlib.pyplot as plt  # 可视化模块
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

plt.figure()
plt.scatter(iris_X[:, 0], iris_X[:, 1], marker='v',
            c=iris_Y, cmap=plt.cm.gnuplot)

x_min, x_max = iris_X[:, 0].min() - 0.5, iris_X[:, 0].max() + 0.5
y_min, y_max = iris_X[:, 1].min() - 0.5, iris_X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                     np.arange(y_min, y_max, 0.2))

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())


plt.show()

# SCATTERPLOT 3D
fig = plt.figure()
ax = Axes3D(fig)
ax.set_title('Iris Dataset by PCA', size=14)
ax.scatter(iris_X[:, 0], iris_X[:, 1], iris_X[:, 2], c=iris_Y)
ax.set_xlabel('First eigenvector')
ax.set_ylabel('Second eigenvector')
ax.set_zlabel('Third eigenvector')
ax.w_xaxis.set_ticklabels(())
ax.w_yaxis.set_ticklabels(())
ax.w_zaxis.set_ticklabels(())
plt.show()


X_train, X_test, Y_train, Y_test = train_test_split(
    iris_X, iris_Y, test_size=0.3, random_state=10)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)


Y_prodeice = knn.predict(X_test)
print(knn.score(X_test, Y_test))

# #保存Model(注:save文件夹要预先建立，否则会报错)
# with open('save/clf.pickle', 'wb') as f:
#     pickle.dump(knn, f)

# #读取Model
# with open('save/clf.pickle', 'rb') as f:
#     clf2 = pickle.load(f)
#     #测试读取后的Model
#     print(clf2.predict(iris_X[0:10]))

joblib.dump(knn, 'save/knn.pkl')

# 读取Model
clf3 = joblib.load('save/knn.pkl')

# 测试读取后的Model
print(clf3.predict(iris_X[0:1]))
print(clf3.score(X_test, Y_test))


scores = cross_val_score(knn, iris_X, iris_Y, cv=5, scoring='accuracy')

print(scores)


# 建立测试参数集
k_range = range(1, 31)

k_scores = []

# 藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = -cross_val_score(knn, iris_X, iris_Y,
                              cv=10, scoring='neg_mean_squared_error')
    k_scores.append(scores.mean())

# 可视化数据
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
