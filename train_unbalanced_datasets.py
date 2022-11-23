import torch
import os
from config import Config
from utils import load_data_dude, result_dude, save_evaluation_dude, split_data
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

    AUC, AUPR, F1_score, LOSS = result_dude(total_loss, pred_labels, predictions, labels)
    return AUC, AUPR, F1_score, LOSS


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

    AUC, AUPR, F1_score, LOSS = result_dude(total_loss, pred_labels, predictions, labels)
    return AUC, AUPR, F1_score, LOSS


def run(task, model, optimizer, train_data, valid_data, test_data_11, test_data_13, test_data_15):
    count_decline = 0
    save_path = './output/interaction/{}/valid_best_checkpoint.pth'.format(task)
    valid_max_AUC = 0
    res = []
    for epoch in range(1, 50):
        print('****** epoch:{} ******'.format(epoch))

        model.train()
        AUC, AUPR, F1_score, LOSS = train_model(model, optimizer, train_data)
        res1 = [epoch, AUC, AUPR, F1_score, LOSS]
        save_evaluation_dude(task, "train", res1)
        print("Train>>> AUC:%f  AUPR:%f   F1_score:%f  LOSS:%f" % (AUC, AUPR, F1_score, LOSS))

        model.eval()
        AUC, AUPR, F1_score, LOSS = test_model(model, valid_data)
        res2 = [epoch, AUC, AUPR, F1_score, LOSS]
        save_evaluation_dude(task, "valid", res2)
        print("Valid>>> AUC:%f  AUPR:%f  F1_score:%f  LOSS:%f" % (AUC, AUPR, F1_score, LOSS))

        AUC, AUPR, F1_score, LOSS = test_model(model, test_data_11)
        res3 = [epoch, AUC, AUPR, F1_score, LOSS]
        save_evaluation_dude(task, "test_11", res3)
        print("Test_11>>> AUC:%f  AUPR:%f  F1_score:%f LOSS:%f" % (AUC, AUPR, F1_score, LOSS))

        AUC, AUPR, F1_score, LOSS = test_model(model, test_data_13)
        res4 = [epoch, AUC, AUPR, F1_score, LOSS]
        save_evaluation_dude(task, "test_13", res4)
        print("Test_13>>> AUC:%f  AUPR:%f  F1_score:%f LOSS:%f" % (AUC, AUPR, F1_score, LOSS))

        AUC, AUPR, F1_score, LOSS = test_model(model, test_data_15)
        res5 = [epoch, AUC, AUPR, F1_score, LOSS]
        save_evaluation_dude(task, "test_15", res5)
        print("Test_15>>> AUC:%f AUPR:%f  F1_score:%f LOSS:%f" % (AUC, AUPR, F1_score, LOSS))

        if res2[1] > valid_max_AUC:
            valid_max_AUC = res2[1]
            res = res3
            count_decline = 0
            torch.save(model.state_dict(), save_path)
        else:
            count_decline = count_decline + 1
            if count_decline >= 15:
                print("EarlyStopping !!!")
                break

        if epoch % 10 == 0:       # 每隔10次学习率衰减一次
            optimizer.param_groups[0]['lr'] *= config.lr_decay
    return res


if __name__ == "__main__":
    tasks = ['dude_1_1', 'dude_1_3', 'dude_1_5']
    methods_components = 'icmf'
    l_drugs_dict = len(pickle.load(open('./datasets/interaction/' + tasks[2] + '/drugs_dict', 'rb')))
    l_proteins_dict = len(pickle.load(open('./datasets/interaction/' + tasks[2] + '/proteins_dict', 'rb')))
    if methods_components == 'icmf':
        l_dict = len(pickle.load(open('./datasets/interaction/' + tasks[2] + '/drugs_cmf_dict', 'rb')))
    else:
        l_dict = 23532

    model = MdDTI(l_drugs_dict, l_proteins_dict, l_dict)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=config.weight_decay)

    train_valid_data = load_data_dude(tasks[0], 'train', methods_components)
    test_data_11 = load_data_dude(tasks[0], 'test', methods_components)
    test_data_13 = load_data_dude(tasks[1], 'test', methods_components)
    test_data_15 = load_data_dude(tasks[2], 'test', methods_components)
    train_data, valid_data = split_data(train_valid_data, 0.1)

    run('dude', model, optimizer, train_data, valid_data, test_data_11, test_data_13, test_data_15)
