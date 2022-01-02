import matplotlib
import numpy as np

import MnistDataLoader
import numpy
import numpy.linalg as la
import matplotlib.pyplot as plt

import sklearn.cluster


def decrease_dimension(x, p, u):
    w = (u.transpose() @ x)
    return w[:p]

def show_back_as_image(x,u,p = 20):
    new_img = u @ x
    plt.imshow(new_img.reshape(28,28), cmap='gray')
    plt.title(f'image in rank {[p]}')
    plt.show()



def find_closest_center_index(centers,img):
    closest_idx = 0
    dist = ((img - centers[0])**2).sum()
    for i in range(0,10):
        curr_dist = ((img - centers[i])**2).sum()
        if dist > curr_dist:
            closest_idx = i
            dist = curr_dist
    return closest_idx


def compute_covariance_eigendecomposition(x_train):
    # this function compute the decomposition of the covariance matrix
    COV_matrix = (x_train @ (x_train.transpose()))
    COV_matrix *= (1 / len(x_train[0]))
    eigen_values, eigans_vectors_matrix = la.eig(COV_matrix)
    eigen_values = eigen_values.real
    eigans_vectors_matrix = eigans_vectors_matrix.real
    # ploting the eigen values to see that they are decaying:
    singular_values = numpy.array(numpy.square(eigen_values))
    x = [i for i in range(len(singular_values))]
    plt.plot(x,singular_values)
    plt.title('Singular Value Decaying')
    plt.show()
    return eigans_vectors_matrix[:,:20]



def kmeansAfterPca(k, x_imgs, rank, centers=None):
    if centers is None:
        centers = np.random.random((k, rank)) - 0.5
    # num_itr = 0
    changed = True
    clusters = np.empty([k], dtype=object)
    indicators = np.zeros(len(x_imgs)) - 1
    while changed:
        for j in range(k):
            clusters[j] = list()
        changed = False
        # divide to clusters
        for index in range(len(x_imgs)):
            img = x_imgs[index]
            closest_center = 0
            closest_dist = ((img - centers[0])**2).sum()
            for i in range(1, k):
                dist = ((img - centers[i])**2).sum()
                if dist < closest_dist:
                    closest_center = i
                    closest_dist = dist
            clusters[closest_center].append(img)
            if closest_center != indicators[index]:
                changed = True
                indicators[index] = closest_center
        # update the centers
        for i in range(len(centers)):
            center = np.zeros(rank)
            if len(clusters[i]) > 0:
                for img1 in clusters[i]:
                    center += img1
                center = center / len(clusters[i])
                centers[i] = center
            else:
                centers[i] = np.random.random(rank) - 0.5
    return centers, indicators, clusters


def clusterToDigit(indicators,y_train):
    digits_in_each_center = numpy.zeros((10,10))
    for i in range(len(indicators)):
        real_lable = int(y_train[i])
        center = int(indicators[i])
        digits_in_each_center[real_lable][center] += 1
    digits_in_each_center = digits_in_each_center.astype(int).transpose()
    cluster_to_digit = numpy.zeros(10)
    for i in range(10):
        max = 0
        digit = 0
        for j in range(10):
            if digits_in_each_center[i][j] > max:
                max = digits_in_each_center[i][j]
                digit = j
        cluster_to_digit[i] = digit
    return cluster_to_digit







if __name__ == '__main__':
    loader = MnistDataLoader.MnistDataloader('train-images.idx3-ubyte', 'train-labels.idx1-ubyte'
                                             , 't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
    (x_train, y_train), (x_test, y_test) = loader.load_data()
    eigans_vectors_matrix = compute_covariance_eigendecomposition(x_train)
    # decreasing the rank of all images to p = 20
    x_train_new = numpy.array([decrease_dimension(x, 20, eigans_vectors_matrix) for x in x_train.transpose()])
    x_test_new = numpy.array([decrease_dimension(x, 20, eigans_vectors_matrix) for x in x_test.transpose()])
    # showing example of 1 image reducing dimension
    # plt.imshow(x_train.transpose()[4542].reshape(28,28), cmap='gray')
    plt.title('original rank')
    plt.show()
    #new rank
    show_back_as_image(x_train_new[4542],eigans_vectors_matrix,20)
    # #end of section 2 (b),end of code

    centers, indicators, clusters = kmeansAfterPca(10,x_train_new,20)
    cluster_to_digit = clusterToDigit(indicators,y_train)
    successes = 0
    for i in range(len(y_test)):
        center_idx = find_closest_center_index(centers,x_test_new[i])
        if cluster_to_digit[center_idx] == y_test[i]:
            successes += 1
    print(f'percentage of sucess: {(successes/len(y_test))*100}%')


    #repeating the experiment using p = 12
    x_train_new = numpy.array([decrease_dimension(x, 12, eigans_vectors_matrix) for x in x_train.transpose()])
    x_test_new = numpy.array([decrease_dimension(x, 12, eigans_vectors_matrix) for x in x_test.transpose()])
    centers, indicators, clusters = kmeansAfterPca(10, x_train_new, 12)
    cluster_to_digit = clusterToDigit(indicators,y_train)
    successes = 0
    for i in range(len(y_test)):
        center_idx = find_closest_center_index(centers,x_test_new[i])
        if cluster_to_digit[center_idx] == y_test[i]:
            successes += 1
    print(f'p=12 percentage of sucess: {(successes / len(y_test)) * 100}%')


    #Kmeans after initializing each of the Kmeans centroids using
    # the mean of 10 reduced images  from each label.
    x_train_new = numpy.array([decrease_dimension(x, 20, eigans_vectors_matrix) for x in x_train.transpose()])
    x_test_new = numpy.array([decrease_dimension(x, 20, eigans_vectors_matrix) for x in x_test.transpose()])
    # smart_centers = get_centers(20,10,y_train,x_train_new,eigans_vectors_matrix)
    smart_centers = [x_train_new[1],x_train_new[3],x_train_new[5],x_train_new[7],x_train_new[2],x_train_new[0],x_train_new[13],x_train_new[15],x_train_new[17],x_train_new[19]]
    centers, indicators, clusters = kmeansAfterPca(10, x_train_new, 20,smart_centers)
    cluster_to_digit = clusterToDigit(indicators,y_train)
    # for img in centers:
    #     show_back_as_image(img,eigans_vectors_matrix,20)
    successes = 0
    for i in range(len(y_test)):
        center_idx = find_closest_center_index(centers,x_test_new[i])
        if cluster_to_digit[center_idx] == y_test[i]:
            successes += 1
    print(f'percentage of sucess after smart center initialization: {(successes / len(y_test)) * 100}%')




