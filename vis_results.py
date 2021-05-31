import numpy as np
import matplotlib

matplotlib.use('AGG')  #
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #
    method = 'A-GEM'
    dataset = 'MNIST'
    hidden_l4 = np.load('/home/wangshuai/project/RACL/softmaxVis/{}_{}_runid_0_hidden_trainres.npy'.format(method, dataset))
    labels_l4 = np.load('/home/wangshuai/project/RACL/softmaxVis/{}_{}_runid_0_labels_trainres.npy'.format(method, dataset))

    plt.set_cmap('hsv')

    # plt.subplot(244)
    plt.subplot(111)
    print(hidden_l4.shape)
    print(labels_l4.shape)

    color = ['black', 'dimgray', 'silver', 'rosybrown', 'lightcoral', 'darkorange', 'burlywood', 'navajowhite', 'gold',
             'darkkhaki']
    if dataset == 'CIFAR':
        # <editor-fold desc=old draw>
        # labels = list(set(labels_l4.reshape(1500).tolist()))
        # m4 = plt.scatter(hidden_l4[:500, 0], hidden_l4[:500, 1], c=labels_l4.reshape(1500)[:500], label=labels)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
        # </editor-fold>
        hidden_l4 = hidden_l4[:500]
        labels_l4 = labels_l4[:500].reshape(500)
        print(labels_l4.shape)
        for labels in list(set(labels_l4)):
            idx = labels_l4 == labels
            plt.scatter(hidden_l4[idx, 0], hidden_l4[idx, 1])
        plt.legend(np.arange(5, dtype=np.int32))
    elif dataset == 'MNIST':
        # <editor-fold desc=old draw>
        # labels = list(set(labels_l4.reshape(30000).tolist()))
        # m4 = plt.scatter(hidden_l4[:10000, 0], hidden_l4[:10000, 1], c=labels_l4.reshape(30000)[:10000], label=labels) # c = color can't
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
        # </editor-fold>
        hidden_l4 = hidden_l4[:10000]
        labels_l4 = labels_l4[:10000]
        print(hidden_l4.shape)
        for labels in range(10):
            idx = labels_l4 == labels
            # print(idx)
            idx = np.array(idx).reshape(10000)
            plt.scatter(hidden_l4[idx, 0], hidden_l4[idx, 1])
        plt.legend(np.arange(10, dtype=np.int32))

    # plt.show()
    plt.savefig('softmaxVis/vis_{}_{}_trainres.png'.format(method, dataset), dpi=600, bbox_inches='tight')
