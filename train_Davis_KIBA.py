import torch
import os
from config import Config
from utils import load_data_interaction, save_evaluation_Davis_KIBA, split_data, shuffle_dataset, result_Davis
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
    Acc, Pre, Recall, AUC, AUPR, LOSS = result_Davis(total_loss, pred_labels, predictions, labels)
    return Acc, Pre, Recall, AUC, AUPR, LOSS


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

    Acc, Pre, Recall, AUC, AUPR, LOSS = result_Davis(total_loss, pred_labels, predictions, labels)
    return Acc, Pre, Recall, AUC, AUPR, LOSS


def run(task, model, optimizer, train_data, valid_data, test_data):
    count_decline = 0
    save_path = './output/interaction/{}/valid_best_checkpoint.pth'.format(task)
    valid_max_AUC = 0
    res = []
    for epoch in range(1, 150):
        print('****** epoch:{} ******'.format(epoch))

        model.train()
        Acc, Pre, Recall, AUC, AUPR, LOSS = train_model(model, optimizer, train_data)
        res1 = [epoch, Acc, Pre, Recall, AUC, AUPR, LOSS]
        save_evaluation_Davis_KIBA(task, "train", res1)
        print("Train>>> Acc: %f Pre: %f Recall: %f AUC: %f AUPR: %f LOSS:%f" % (Acc, Pre, Recall, AUC, AUPR, LOSS))

        model.eval()
        Acc, Pre, Recall, AUC, AUPR, LOSS = test_model(model, valid_data)
        res2 = [epoch, Acc, Pre, Recall, AUC, AUPR, LOSS]
        save_evaluation_Davis_KIBA(task, "valid", res2)
        print("Valid>>> Acc: %f Pre: %f Recall: %f AUC: %f AUPR: %f LOSS:%f" % (Acc, Pre, Recall, AUC, AUPR, LOSS))

        Acc, Pre, Recall, AUC, AUPR, LOSS = test_model(model, test_data)
        res3 = [epoch, Acc, Pre, Recall, AUC, AUPR, LOSS]
        save_evaluation_Davis_KIBA(task, "test", res3)
        print("Test>>> Acc: %f Pre: %f Recall: %f AUC: %f AUPR: %f LOSS:%f" % (Acc, Pre, Recall, AUC, AUPR, LOSS))

        if res2[4] > valid_max_AUC:
            valid_max_AUC = res2[4]
            res = res3
            count_decline = 0
            torch.save(model.state_dict(), save_path)
        else:
            count_decline = count_decline + 1
            if count_decline >= 20:
                print("EarlyStopping !!!")
                break

        if epoch % 15 == 0:       # 每隔n次学习率衰减一次
            optimizer.param_groups[0]['lr'] *= config.lr_decay
    return res


if __name__ == "__main__":
    tasks = ['Davis', 'KIBA']
    methods_components = 'icmf'
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

        model = MdDTI(l_drugs_dict, l_proteins_dict, l_dict)
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=config.weight_decay)

        train_valid_data, test_data = split_data(train_val_test_data, 0.2)
        train_data, valid_data = split_data(train_valid_data, 0.1)

        res = run(task, model, optimizer, train_data, valid_data, test_data)
        print("Result>>> epoch:%f  Acc: %f Pre: %f Recall: %f AUC: %f AUPR: %f" % (res[0], res[1], res[2], res[3], res[4], res[5]))
