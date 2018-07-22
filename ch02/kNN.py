from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import zipfile


def create_dataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = np.array(['A', 'A', 'B', 'B'])
    return group, labels


def plot_dataset(features, labels):
    fig, ax = plt.subplots(figsize=(6, 6))
    classa = features[labels == 1]
    classb = features[labels == 2]
    classc = features[labels == 3]

    ax.scatter(classa[:, 0], classa[:, 1], label='A', marker='o')
    ax.scatter(classb[:, 0], classb[:, 1], label='B', marker='x')
    ax.scatter(classc[:, 0], classc[:, 1], label='c', marker='*')

    plt.legend(loc='best')
    # plt.show()


def classfiy0(inx, dataset, labels, k):
    diffmat = dataset - inx
    l2d = np.sqrt(np.sum(np.square(diffmat), axis=1))

    sorted_indices = l2d.argsort()[:k]
    klabels = labels[sorted_indices]

    label_count = dict(zip(*np.unique(klabels, return_counts=True)))

    return max(label_count, key=label_count.get)


def file2matrix(filename):
    dataset = np.fromfile(filename, sep='\t').reshape([-1, 4])
    features = dataset[:, :3]
    labels = dataset[:, 3].astype(np.int32)
    return features, labels


def feature_normalize(features):
    mins = np.min(features, axis=0)
    maxs = np.max(features, axis=0)
    features = (features - mins) / (maxs - mins)
    return features


def dating_class_test():
    train_features, train_labels = file2matrix('./datingTestSet2.txt')
    train_features = feature_normalize(train_features)

    file_conv('./datingTestSet.txt')

    test_features, test_labels = file2matrix('./datingTestSet_conv.txt')
    test_features = feature_normalize(test_features)

    correct_cnt = 0
    for i in range(len(test_features)):
        predict_label = classfiy0(test_features[i], train_features, train_labels, 3)
        # print(predict_label, test_labels[i])
        if(predict_label == test_labels[i]):
            correct_cnt += 1

    print(correct_cnt/len(test_features))


def file_conv(filename):
    with open(filename, 'r') as f:
        content = f.read()
        content = content.replace('largeDoses', '3')
        content = content.replace('smallDoses', '2')
        content = content.replace('didntLike', '1')

    with open('datingTestSet_conv.txt', 'w') as f:
        f.write(content)


def img2vector(zipfp, filename):
    img_str = zipfp.read(filename).decode('utf-8')
    img_lines = img_str.split('\r\n')
    img = [int(ch) for line in img_lines for ch in line]
    return np.array(img)


def zipfile_extract(filename):

    with zipfile.ZipFile(filename, 'r') as f:
        filelist = f.namelist()

        train_filelist = [x for x in filelist if x.split('/')[0] == "trainingDigits"][1:]
        test_filelist = [x for x in filelist if x.split('/')[0] == "testDigits"][1:]

        train_features = np.array([img2vector(f, x) for x in train_filelist])
        test_features = np.array([img2vector(f, x) for x in test_filelist])

        train_labels = np.array([int(item.split('_')[0][-1]) for item in train_filelist])
        test_labels = np.array([int(item.split('_')[0][-1]) for item in test_filelist])

    return train_features, train_labels, test_features, test_labels


def img_class_test():
    train_x, train_y, test_x, test_y = zipfile_extract('digits.zip')
    # plt.imshow(train_x[0].reshape(32, 32), cmap='gray')
    # plt.show()

    correct_cnt = 0
    for i in range(len(test_x)):
        predict_label = classfiy0(test_x[i], train_x, train_y, 3)
        # print(predict_label, test_y[i])
        if(predict_label == test_y[i]):
            correct_cnt += 1

    print(correct_cnt/len(test_x))


if __name__ == '__main__':
    group, labels = create_dataset()
    inx = np.array([0.0, 0.2])
    klabels = classfiy0(inx, group, labels, 3)
    print(klabels)
    dating_class_test()
    img_class_test()

