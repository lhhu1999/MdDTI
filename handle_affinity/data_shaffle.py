import numpy as np
import warnings
warnings.filterwarnings("ignore")


def shuffle_dataset(data):
    np.random.seed(1234)
    np.random.shuffle(data)
    return data


if __name__ == '__main__':
    datasets = ['Kd', 'EC50']

    for dataset in datasets:
        for filename in ['train', 'test']:
            fpath = '../RawData/affinity/{}/{}.txt'.format(dataset, filename)

            with open(fpath, "r") as f:
                data_list = f.read().strip().split('\n')
            f.close()

            data_shuffle = shuffle_dataset(data_list)

            output = "../RawData/affinity/{}/{}_shuffle.txt".format(dataset, filename)
            with open(output, 'w') as f:
                for i in data_shuffle:
                    f.write(str(i) + '\n')
                f.close()
        print("shuffle succeeded in " + dataset)
    print("All succeeded !!!")
