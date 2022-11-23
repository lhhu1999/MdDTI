import torch
import os
from config import Config
from utils import load_data_interaction, result_interaction, save_evaluation_interaction, get_kfold_data, split_data, shuffle_dataset
from model import MdDTI
import math
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import pickle
import warnings
warnings.filterwarnings("ignore")

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

config = Config()
if config.mode == 'gpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print("cuda is not available!!!")
        device = torch.device('cpu')
else:
    device = torch.device('cpu')
print('>>>>>> The code run on the ', device)
criterion = nn.CrossEntropyLoss()


def train_model(model, optimizer, train_data):
    lens = len(train_data)
    train_len = np.arange(len(train_data[0]))
    counts = math.floor(len(train_data[0]) / config.batch_size)   # 向上取整
    pred_labels = []
    predictions = []
    labels = []
    total_loss = []
    for i in tqdm(range(counts)):
        optimizer.zero_grad()
        batch_data = [train_data[j][train_len[i * config.batch_size: (i + 1) * config.batch_size]] for j in range(lens)]
        out, loss2 = model(batch_data, device, 'interaction')

        label = torch.FloatTensor(batch_data[5]).to(device)

        loss = criterion(out.float(), label.reshape(label.shape[0]).long()) + loss2
        total_loss.append(loss.cpu().detach())
        ys = F.softmax(out, 1).to('cpu').data.numpy()
        pred_labels += list(map(lambda x: np.argmax(x), ys))  # (32)预测结果
        predictions += list(map(lambda x: x[1], ys))  # (32)标签为1的值
        labels += label.cpu().numpy().reshape(-1).tolist()

        loss.backward()
        optimizer.step()

    AUC, Pre, Recall, LOSS = result_interaction(total_loss, pred_labels, predictions, labels)
    return AUC, Pre, Recall, LOSS


def test_model(model, test_data):
    lens = len(test_data)
    test_len = np.arange(len(test_data[0]))
    counts = math.floor(len(test_data[0]) / config.batch_size)
    pred_labels = []
    predictions = []
    labels = []
    total_loss = []
    with torch.no_grad():
        for i in tqdm(range(counts), colour='blue'):
            batch_data = [test_data[j][test_len[i * config.batch_size: (i + 1) * config.batch_size]] for j in range(lens)]
            out, loss2 = model(batch_data, device, 'interaction')

            label = torch.FloatTensor(batch_data[5]).to(device)

            loss = criterion(out.float(), label.reshape(label.shape[0]).long()) + loss2
            total_loss.append(loss.cpu().detach())
            ys = F.softmax(out, 1).to('cpu').data.numpy()
            pred_labels += list(map(lambda x: np.argmax(x), ys))  # (32)预测结果
            predictions += list(map(lambda x: x[1], ys))  # (32)标签为1的值
            labels += label.cpu().numpy().reshape(-1).tolist()

    AUC, Pre, Recall, LOSS = result_interaction(total_loss, pred_labels, predictions, labels)
    return AUC, Pre, Recall, LOSS


def run(task, k, model, optimizer, train_data, valid_data, test_data):
    save_path = './output/interaction/{}/{}_valid_best_checkpoint.pth'.format(task, k)
    valid_max_AUC = 0
    res = []
    for epoch in range(1, 40):
        print('****** epoch:{} ******'.format(epoch))

        model.train()
        AUC, Pre, Recall, LOSS = train_model(model, optimizer, train_data)
        res1 = [epoch, AUC, Pre, Recall, LOSS]
        save_evaluation_interaction(task, "train", k, res1)
        print("Train>>> AUC:%f  Pre:%f  Recall:%f  LOSS:%f" % (AUC, Pre, Recall, LOSS))

        model.eval()
        AUC, Pre, Recall, LOSS = test_model(model, valid_data)
        res2 = [epoch, AUC, Pre, Recall, LOSS]
        save_evaluation_interaction(task, "valid", k, res2)
        print("Valid>>> AUC:%f  Pre:%f  Recall:%f  LOSS:%f" % (AUC, Pre, Recall, LOSS))

        AUC, Pre, Recall, LOSS = test_model(model, test_data)
        res3 = [epoch, AUC, Pre, Recall, LOSS]
        save_evaluation_interaction(task, "test", k, res3)
        print("Test>>> AUC:%f  Pre:%f  Recall:%f LOSS:%f" % (AUC, Pre, Recall, LOSS))

        if res2[1] > valid_max_AUC:
            valid_max_AUC = res2[1]
            res = res3
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0:       # 每隔10次学习率衰减一次
            optimizer.param_groups[0]['lr'] *= config.lr_decay
    return res


if __name__ == "__main__":
    K_FOLD = config.K_FOLD
    tasks = ['celegans', 'human']
    methods_components = 'bpe'

    for task in tasks:
        print('********************** train in {} **********************'.format(task))
        l_drugs_dict = len(pickle.load(open('./datasets/interaction/' + task + '/drugs_dict', 'rb')))
        l_proteins_dict = len(pickle.load(open('./datasets/interaction/' + task + '/proteins_dict', 'rb')))
        if methods_components == 'icmf':
            l_dict = len(pickle.load(open('./datasets/interaction/' + task + '/drugs_cmf_dict', 'rb')))
        else:
            l_dict = 23532

        train_val_test_data = load_data_interaction(task, methods_components)
        train_val_test_data = shuffle_dataset(train_val_test_data, SEED)

        for k in range(4, K_FOLD + 1):
            model = MdDTI(l_drugs_dict, l_proteins_dict, l_dict)
            model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=config.weight_decay)

            train_valid_data, test_data = get_kfold_data(k, train_val_test_data, K_FOLD)
            train_data, valid_data = split_data(train_valid_data, 0.1)

            res = run(task, k, model, optimizer, train_data, valid_data, test_data)
            print("*************** %d_fold ***************" % k)
            print("Result: >>> epoch:%f  AUC:%f  Pre:%f  Recall:%f" % (res[0], res[1], res[2], res[3]))
