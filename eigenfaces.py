import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import itertools

from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

plt.rcParams['image.cmap'] = "gray"

# Code adapted from scikit-learn docs
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure()
    fig.set_figheight(20)
    fig.set_figwidth(20)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm2 = cm

    thresh = cm.max() * 0.50
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, int(cm2[i, j]*100+0.5),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig('cm.png')

def show_misclassified(row_total, col_total, test, pred=-1, notpred=False):
    fig = plt.figure()
    fig.set_figheight(4*row_total)
    fig.set_figwidth(3*col_total)
    # set pred = -1 to show all faces from a person from test
    # set notpred = True to set filter to NOT equal pred
    if pred == -1:
        print('All faces of ' + people.target_names[test] + ':')
        filter = (y_test==test)
    elif notpred:
        if test == pred:
            print('Faces of ' + people.target_names[test] + ' misclassified:')
        else:
            print('Faces of ' + people.target_names[test] + ' NOT classifed as ' + people.target_names[pred] + ':')
        filter = (y_test==test) & (y_pred != pred)
    else:
        if test == pred:
            print('Faces of ' + people.target_names[test] + ' classified correctly:')
        else:
            print('Faces of ' + people.target_names[test] + ' misclassified as ' + people.target_names[pred] + ':')
        filter = (y_test==test) & (y_pred == pred)
    misclassified = X_test[filter]
    total = sum(filter)
    for row, col in itertools.product(range(row_total), range(col_total)):
        ind = row*col_total + col
        if ind < total:
            ax = plt.subplot2grid((row_total, col_total), (row, col))
            ax.axis('off')
            ind = row*col_total + col
            ax.set_title(people.target_names[test])
            ax.imshow(misclassified[ind].reshape(87,65),cmap='gray')
    plt.tight_layout()

#fetch faces from LFW
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    imshow = ax.imshow(image, cmap=None)
    ax.set_title(people.target_names[target])

print("people.images.shape: {}".format(people.images.shape))
print("Number of classes: {}".format(len(people.target_names)))

# count how often each target appears
counts = np.bincount(people.target)
# print counts next to target names:
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='   ')
    if (i + 1) % 3 == 0:
        print()
# data too skewed, take 50 per person
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

# scale the grey-scale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability:
X_people = X_people / 255

# split the data in training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0)
# build a KNeighborsClassifier with using one neighbor:
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test, y_test)))

# 1-nn is about 27% accurate - use PCA


pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("X_train_pca.shape: {}".format(X_train_pca.shape))

# 100 features / principal components



# compare knn with svc

print("pca.components_.shape: {}".format(pca.components_.shape))

fig, axes = plt.subplots(3, 5, figsize=(15, 12),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape),
              cmap='viridis')
    ax.set_title("{}. component".format((i + 1)))
fig.savefig('pca.png')
#plt.show()
k = np.arange(1, 15, 1)
score = np.zeros(14)
print("k    acc.\n")
for i in range(len(k)):
    knn = KNeighborsClassifier(n_neighbors=k[i], weights='distance')
    knn.fit(X_train_pca, y_train)
    score[i] = knn.score(X_test_pca, y_test)
    print("{:2d}   {:.4f}".format(k[i], knn.score(X_test_pca, y_test)), end='   \n')
fig = plt.figure()
plt.plot(k, score, '-')
plt.xlabel('k')
plt.ylabel('Accuracy')
fig.savefig('knn.png')

c = [1,3,5,10,30, 50,100]
gamma = [0.0001, 0.0003, 0.001, 0.003, 0.01]
score = np.zeros((7,5))
print("  c    .0001    .0003    .001     .003     .01\n")
for i in range(len(c)):
    print("{:3d}".format(c[i]), end='   ')
    for j in range(len(gamma)):
        svc = SVC(C=c[i], gamma=gamma[j])
        svc.fit(X_train_pca, y_train)
        score[i][j] = svc.score(X_test_pca, y_test)
        print("{:.4f}".format(score[i][j]), end='   ')
    print("\n")
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('C')
ax.set_ylabel('Gamma')
ax.set_zlabel('Accuracy')

# svc with c=100 and gaamma = 0.01
y_pred=svc.predict(X_test_pca)

c, gamma = np.meshgrid(c, gamma)
# Plot the surface.
surf = ax.plot_surface(c , gamma, np.transpose(score), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0,1)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
fig.savefig('svc.png')

plt.close()
cm=confusion_matrix(people.target_names[y_test], people.target_names[y_pred], labels=people.target_names)
plot_confusion_matrix(cm,people.target_names,normalize=True)
print(classification_report(y_test, y_pred, target_names=people.target_names))


show_misclassified(4, 7, 35, 7)
show_misclassified(4, 7, 35, 7, True)
show_misclassified(4, 7, 7)

show_misclassified(4, 7, 35, 35)

show_misclassified(4, 5, 33, 42)
show_misclassified(4, 5, 33, 42, True)
show_misclassified(4, 7, 42)

show_misclassified(4, 5, 33, 33)

show_misclassified(4, 5, 2, 23)
show_misclassified(4, 5, 2, 23, True)
show_misclassified(4, 7, 23)

show_misclassified(4, 5, 2, 2)

cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
num = np.zeros(45)
for i in range(45):
    num[i] = len(X_test[(y_test==i)])
fig = plt.figure()
plt.plot(num, np.diag(cm2), 'o')
plt.title('Number of Faces and Accuracy')
plt.xlabel('Number of faces')
plt.ylabel('Accuracy')

