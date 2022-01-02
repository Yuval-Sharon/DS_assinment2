import matplotlib
import numpy as np

import MnistDataLoader
import numpy
import numpy.linalg as la
import matplotlib.pyplot as plt

import sklearn.cluster


def decrease_dimension(x, p, u):
    # this function reduce the dimension of (image) x to p.
    new_img = numpy.zeros(len(x))
    w = (u.transpose() @ x)
    # for i in range(p):
    #     new_img += u.transpose()[i].real * w[i].real
    #
    # return new_img
    return w[:p]

def show_back_as_image(x,u,p,digit):
    new_img = numpy.zeros(784)
    for i in range(p):
         new_img += u.transpose()[i].real * x[i].real
    plt.imshow(new_img.reshape(28,28), cmap='gray')
    plt.title(f'center represent digit -{digit}')
    plt.show()



def find_closest_center_index(centers,img):
    closest_idx = 0
    dist = ((img - centers[0])**2).sum()
    for i in range(0,10):
        curr_dist = ((img - centers[i])**2).sum()
        if dist > curr_dist:
            closest_idx = i
            dist = curr_dist
    # print(closest_idx)
    return closest_idx


def compute_covariance_eigendecomposition(x_train):
    # this function compute the eigendecomposition of the covariance matrix
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


def kmeans(k, x_imgs,centers = np.random.random((10,784))-0.5):
    num_itr = 0
    # x_imgs = x_imgs.transpose()
    changed = True
    clusters = np.empty([10], dtype=object)
    indicators = np.zeros(len(x_imgs)) -1
    while changed:
        for j in range(10):
            clusters[j] = list()
        num_itr += 1
        print(num_itr)
        changed = False
        #divide to clusters
        for index in range(len(x_imgs)):
            img = x_imgs[index]
            closest_center = 0
            closest_dist = ((img - centers[0])**2).sum()
            for i in range(1,10):
                dist = ((img - centers[i])**2).sum()
                if dist < closest_dist:
                    closest_center = i
                    closest_dist = dist
            clusters[closest_center].append(img)
            if closest_center != indicators[index]:
                changed = True
                indicators[index] = closest_center
        for i in range(len(centers)):
            center = np.zeros(784)
            if len(clusters[i]) > 0:
                for img1 in clusters[i]:
                    center += img1
                center = center / len(clusters[i])
                centers[i] = center
            # centers[i] = np.mean(np.array(clusters[i]),)
            # print(f'center[{i}] = {centers[i]}')
            # print(f'clusters[{i}] = {np.array(clusters[i])}')

    print(f'kmeans iters = {num_itr}')
    return centers,indicators,clusters


def kmeansAfterPca(k, x_imgs, rank, centers=None):
    if (centers == None):
        centers = np.random.random((k, rank)) - 0.5
    # num_itr = 0
    changed = True
    clusters = np.empty([k], dtype=object)
    indicators = np.zeros(len(x_imgs)) - 1
    while changed:
        for j in range(k):
            clusters[j] = list()
        # num_itr += 1
        # print(num_itr)
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


def show_back_as_image_before_pca(img,digit):
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f'center represent digit -{digit}')
    plt.show()


def get_centers(p,k,labels,x_train_in_rank_p):
    smart_centers = np.zeros((k,p))
    found = 0
    digit_found = numpy.zeros(k)
    for j in range(len(labels)):
        if digit_found[labels[j]] == 0:
            digit_found[labels[j]] = 1
            found +=1
            smart_centers[labels[j]] =x_train_in_rank_p[j]
        if found == 9:
            return smart_centers





if __name__ == '__main__':
    # matplotlib.use('Qt5Agg')
    loader = MnistDataLoader.MnistDataloader('train-images.idx3-ubyte', 'train-labels.idx1-ubyte'
                                             , 't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
    (x_train, y_train), (x_test, y_test) = loader.load_data()
    print(x_test.shape)
    print(y_test)
    # x_test = x_test[:6000]
    # y_test = y_test[:6000]
    eigans_vectors_matrix = compute_covariance_eigendecomposition(x_train)
    # showing example of 1 image reducing demension
    # plt.imshow(x_train.transpose()[4542].reshape(28,28), cmap='gray')
    # plt.title('original rank')
    # plt.show()
    # plt.imshow(decrease_dimension(x_train.transpose()[4542]
    #                               ,20,eigans_vectors_matrix).reshape((28,28)),cmap='gray')
    # plt.title('rank = 20')
    # plt.show()
    # decreasing the rank of all images to p = 20
    x_train_new = numpy.array([decrease_dimension(x, 20, eigans_vectors_matrix) for x in x_train.transpose()])
    # x_train_new = x_train
    #end of section 2 (b)
    # numpy.apply_along_axis(decrease_dimension(x,p,eigans_vectors_matrix),1,x_train.transpose())
    # clusters = kmeans(10,x_train_new)[2]
    # for i in range(20):
    #     img = clusters[3][i].reshape((28,28))
    #     plt.imshow(img,cmap='gray')
    #     plt.show()
    centers, indicators, clusters = kmeansAfterPca(10,x_train_new,20)
    # centers, indicators, clusters = kmeans(10,x_train.transpose())
    cluster_to_digit = clusterToDigit(indicators,y_train)
    successes = 0
    # for k in range(10):
    #     show_back_as_image(centers[k],eigans_vectors_matrix,20,cluster_to_digit[k])
    #     # show_back_as_image_before_pca(centers[k],cluster_to_digit[k])
    # # x_test_new = x_test
    x_test_new = numpy.array([decrease_dimension(x, 20, eigans_vectors_matrix) for x in x_test.transpose()])
    print(len(y_test))
    for i in range(len(y_test)):
        center_idx = find_closest_center_index(centers,x_test_new[i])
        if cluster_to_digit[center_idx] == y_test[i]:
            successes += 1
    # print(f'sucess {successes}')
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
    x_test_new = numpy.array([decrease_dimension(x, 12, eigans_vectors_matrix) for x in x_test.transpose()])
    smart_centers = get_centers(20,10,y_train,x_train_new)
    centers, indicators, clusters = kmeansAfterPca(10, x_train_new, 20,smart_centers)
    cluster_to_digit = clusterToDigit(indicators,y_train)
    successes = 0
    for i in range(len(y_test)):
        center_idx = find_closest_center_index(centers,x_test_new[i])
        if cluster_to_digit[center_idx] == y_test[i]:
            successes += 1
    print(f' percentage of sucess after smart center initialization: {(successes / len(y_test)) * 100}%')






# todo :we can delete all this - because we need to implement our own version of Kmeans

# labels,centers = (using_Kmeans(10, x_train_new))
# print(labels.size)
# digits_in_each_center = numpy.zeros((10,10))
# for i in range(len(labels)):
#     real_lable = y_train[i]
#     center = labels[i]
#     digits_in_each_center[real_lable][center] += 1
# digits_in_each_center = digits_in_each_center.astype(int).transpose()
# cluster_to_digit = numpy.zeros(10)
# for i in range(10):
#     max = 0
#     digit = 0
#     for j in range(10):
#         if digits_in_each_center[i][j] > max:
#             max = digits_in_each_center[i][j]
#             digit = j
#     cluster_to_digit[i] = digit
# print(cluster_to_digit)
# sucesses = 0
# x_test = x_test.transpose()
# print(centers)
# for i in range(len(y_test)):
#     center_idx = find_closest_center_index(centers,x_test[i])
#     if center_idx != 2:
#         print(center_idx)
#     if cluster_to_digit[center_idx] == y_test[i]:
#         sucesses += 1
# print(f'percentage of sucess: {sucesses/len(x_test)}')





