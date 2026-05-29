import numpy
import plt


def mcol(v):
    return v.reshape((v.size, 1))

def load(file_name):
    dataset_list = []
    label_list = []
    labels = {
        "Iris-setosa" : 0,
        "Iris-versicolor" : 1,
        "Iris-virginica" : 2
    }

    with open(file_name) as file:
        for line in file:
            try:
                attributes = line.split(",")[0:-1]
                attributes = mcol(numpy.array([float(i) for i in attributes]))
                name = line.split(",")[-1].strip()
                label = labels[name]
                dataset_list.append(attributes)
                label_list.append(label)
            except:
                pass

    return numpy.hstack(dataset_list), numpy.array(label_list, dtype=numpy.int32)

def plot_hist(D, L):

    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    D2 = D[:, L == 2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
    }
    for dIdx in range(4):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.ylabel('Density')
        plt.hist(D0[dIdx, :], bins=10, density=True, alpha=0.4, label='Setosa')
        plt.hist(D1[dIdx, :], bins=10, density=True, alpha=0.4, label='Versicolor')
        plt.hist(D2[dIdx, :], bins=10, density=True, alpha=0.4, label='Virginica')

        plt.legend()
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
        plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()


def plot_scatter(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    D2 = D[:, L == 2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
    }

    for dIdx1 in range(4):
        for dIdx2 in range(4):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label='Setosa')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label='Versicolor')
            plt.scatter(D2[dIdx1, :], D2[dIdx2, :], label='Virginica')

            plt.legend()
            plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
            plt.savefig('scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        plt.show()

if __name__ == "__main__":

    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load('Solution/iris.csv')
    plot_hist(D, L)
    plot_scatter(D, L)

    mu = D.mean(1).reshape((D.shape[0], 1))
    print('Mean:')
    print(mu)
    print()

    DC = D - mu

    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    print('Covariance matrix:')
    print(C)
    print()

    var = D.var(1)
    std = D.std(1)
    print('Variance:', var)
    print('Std. dev.:', std)
    print()

    for cls in [0, 1, 2]:
        print('Class', cls)
        DCls = D[:, L == cls]
        mu = DCls.mean(1).reshape(DCls.shape[0], 1)
        print('Mean:')
        print(mu)
        C = ((DCls - mu) @ (DCls - mu).T) / float(DCls.shape[1])
        print('Covariance:')
        print(C)
        var = DCls.var(1)
        std = DCls.std(1)
        print('Variance:', var)
        print('Std. dev.:', std)
        print()