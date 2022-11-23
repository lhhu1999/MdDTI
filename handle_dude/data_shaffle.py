import numpy as np
import warnings
warnings.filterwarnings("ignore")


def shuffle_dataset(data):
    np.random.seed(1234)
    np.random.shuffle(data)
    return data


if __name__ == '__main__':
    datasets = ['dude_1_1', 'dude_1_3', 'dude_1_5']

    for dataset in datasets:
        train_filename = '../RawData/interaction/{}/dude_train.txt'.format(dataset)
        test_filename = '../RawData/interaction/{}/{}_test.txt'.format(dataset, dataset)
        train_output = '../RawData/interaction/{}/train_shuffle.txt'.format(dataset)
        test_output = '../RawData/interaction/{}/test_shuffle.txt'.format(dataset)

        with open(train_filename, "r") as f:
            data_list = f.read().strip().split('\n')
        f.close()
        data_shuffle = shuffle_dataset(data_list)
        with open(train_output, 'w') as f:
            for i in data_shuffle:
                f.write(str(i) + '\n')
        f.close()

        with open(test_filename, "r") as f:
            data_list = f.read().strip().split('\n')
        f.close()
        data_shuffle = shuffle_dataset(data_list)
        with open(test_output, 'w') as f:
            for i in data_shuffle:
                f.write(str(i) + '\n')
        f.close()

        print("shuffle succeeded in " + dataset)
    print("All succeeded !!!")
