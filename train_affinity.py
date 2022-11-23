import torch
import os
from config import Config
from utils import load_data_affinity, result_affinity, save_evaluation_affinity, split_data
from model import MdDTI
import math
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import pickle
import numpy as np
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
criterion = nn.MSELoss()


def train_model(model, optimizer, train_data):
    lens = len(train_data)
    train_len = np.arange(len(train_data[0]))
    counts = math.floor(len(train_data[0]) / config.batch_size)  # 向上取整
    predictions = []
    labels = []
    total_loss = []
    for i in tqdm(range(counts)):
        optimizer.zero_grad()
        batch_data = [train_data[j][train_len[i * config.batch_size: (i + 1) * config.batch_size]] for j in range(lens)]
        out, loss2 = model(batch_data, device, 'affinity')

        label = torch.FloatTensor(batch_data[5]).to(device)

        loss = criterion(out.float(), label.float())
        total_loss.append(loss.cpu().detach())
        predictions += out.cpu().detach().numpy().reshape(-1).tolist()
        labels += label.cpu().detach().numpy().reshape(-1).tolist()

        optimizer.zero_grad()
        (loss+loss2).backward()
        optimizer.step()

    predictions = np.array(predictions)
    labels = np.array(labels)
    RMSE, PCC, LOSS = result_affinity(total_loss, labels, predictions)

    return RMSE, PCC, LOSS


def test_model(model, test_data):
    lens = len(test_data)
    test_len = np.arange(len(test_data[0]))
    counts = math.floor(len(test_data[0]) / config.batch_size)
    predictions = []
    labels = []
    total_loss = []
    with torch.no_grad():
        for i in tqdm(range(counts), colour='blue'):
            batch_data = [test_data[j][test_len[i * config.batch_size: (i + 1) * config.batch_size]] for j in range(lens)]
            out, loss2 = model(batch_data, device, 'affinity')

            label = torch.FloatTensor(batch_data[5]).to(device)

            loss = criterion(out.float(), label.float())
            total_loss.append(loss.cpu().detach())
            predictions += out.cpu().detach().numpy().reshape(-1).tolist()
            labels += label.cpu().detach().numpy().reshape(-1).tolist()

    predictions = np.array(predictions)
    labels = np.array(labels)
    RMSE, PCC, LOSS = result_affinity(total_loss, labels, predictions)
    return RMSE, PCC, LOSS


def run(task, model, optimizer, train_data, valid_data, test_data):
    count_decline = 0
    save_path = './output/affinity/{}/valid_best_checkpoint.pth'.format(task)
    valid_max_PCC = 0
    res = []
    for epoch in range(1, 70):
        print('****** epoch:{} ******'.format(epoch))

        model.train()
        RMSE, PCC, LOSS = train_model(model, optimizer, train_data)
        res1 = [epoch, RMSE, PCC, LOSS]
        save_evaluation_affinity(task, "train", res1)
        print("Train>>> RMSE: %f PCC: %f LOSS:%f" % (RMSE, PCC, LOSS))

        model.eval()
        RMSE, PCC, LOSS = test_model(model, valid_data)
        res2 = [epoch, RMSE, PCC, LOSS]
        save_evaluation_affinity(task, "valid", res2)
        print("Valid>>> RMSE: %f PCC: %f LOSS:%f" % (RMSE, PCC, LOSS))

        RMSE, PCC, LOSS = test_model(model, test_data)
        res3 = [epoch, RMSE, PCC, LOSS]
        save_evaluation_affinity(task, "test", res3)
        print("Test>>> RMSE: %f PCC: %f LOSS:%f" % (RMSE, PCC, LOSS))

        if res2[2] > valid_max_PCC:
            valid_max_PCC = res2[2]
            res = res3
            count_decline = 0
            torch.save(model.state_dict(), save_path)
        else:
            count_decline = count_decline + 1
            if count_decline == 15:
                print("EarlyStopping !!!")
                break

        if epoch % 10 == 0:       # 每隔n次学习率衰减一次
            optimizer.param_groups[0]['lr'] *= config.lr_decay
    return res


if __name__ == "__main__":
    tasks = ['Kd', 'EC50']
    methods_components = 'icmf'
    for task in tasks:
        print('********************** train in {} **********************'.format(task))
        l_drugs_dict = len(pickle.load(open('./datasets/affinity/' + task + '/drugs_dict', 'rb')))
        l_proteins_dict = len(pickle.load(open('./datasets/affinity/' + task + '/proteins_dict', 'rb')))
        if methods_components == 'icmf':
            l_dict = len(pickle.load(open('./datasets/affinity/' + task + '/drugs_cmf_dict', 'rb')))
        else:
            l_dict = 23532

        model = MdDTI(l_drugs_dict, l_proteins_dict, l_dict)
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=config.weight_decay)

        train_valid_data = load_data_affinity(task, 'train', methods_components)
        test_data = load_data_affinity(task, 'test', methods_components)
        train_data, valid_data = split_data(train_valid_data, 0.1)

        res = run(task, model, optimizer, train_data, valid_data, test_data)
        print("Result: >>> epoch:%f  RMSE: %f PCC: %f" % (res[0], res[1], res[2]))
