import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Sequence
from matplotlib.pyplot import imread
from skimage.transform import resize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from glob import glob


def plot_letters(X, y_true, y_pred=None, n=4, random_state=123):
    np.random.seed(random_state)
    indices = np.random.choice(np.arange(X.shape[0]), size=n*n, replace=False)
    plt.figure(figsize=(10, 10))
    for i in range(n*n):
        plt.subplot(n, n, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[indices[i]].reshape(28, 28), cmap='gray')
        # plt.imshow(train_images[i], cmap=plt.cm.binary)
        if y_pred is None:
            title = chr(ord("A") + y_true[indices[i]])
        else:
            title = f"y={chr(ord('A') + y_true[indices[i]])}, ŷ={chr(ord('A') + y_pred[indices[i]])}"  # noqa
        plt.title(title, size=20)
    plt.show()


def plot_confusion_matrix(
    y_test: Sequence,
    y_test_pred: Sequence,
    model_name: str
) -> None:
    conf_mat: np.ndarray = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(conf_mat, annot=True, fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f"Confusion Matrix for '{model_name}'")
    plt.show()


def load_notmnist(path='./notMNIST_small', letters='ABCDEFGHIJ',
                  img_shape=(28, 28), test_size=0.25, one_hot=False):

    if not os.path.exists(path):
        print("Downloading data...")
        assert os.system('curl http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz > notMNIST_small.tar.gz') == 0  # noqa
        print("Extracting ...")
        assert os.system(
            'tar -zxvf notMNIST_small.tar.gz > untar_notmnist.log'
        ) == 0

    data, labels = [], []
    print("Parsing...")
    for img_path in glob(os.path.join(path, '*/*')):
        class_i = img_path.split(os.sep)[-2]
        if class_i not in letters:
            continue
        try:
            data.append(resize(imread(img_path), img_shape))
            labels.append(class_i,)
        except:  # noqa
            print("found broken img: %s [it's ok if <10 images are broken]" % img_path)  # noqa

    data = np.stack(data)[:, None].astype('float32')
    data = (data - np.mean(data)) / np.std(data)

    # Convert classes to ints
    letter_to_i = {l: i for i, l in enumerate(letters)}
    labels = np.array(list(map(letter_to_i.get, labels)))

    if one_hot:
        labels = (
            np.arange(np.max(labels) + 1)[None, :] == labels[:, None]
        ).astype('float32')

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=test_size,
        stratify=labels
    )

    print("Done")
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = load_notmnist(letters='ABCDEFGHIJ')
X_train, X_test = X_train.reshape([-1, 784]), X_test.reshape([-1, 784])
