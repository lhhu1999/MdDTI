import argparse
import os
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from dataloader import MyDataset, collate_fn
from functools import partial
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
from model import MdDTI
from utils import load_common_data, load_train_data_DTI, get_five_fold_idx
from evaluate import evaluate_DTI, save_result_DTI, save

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

l_drug_dict = len(pickle.load(open('./common/dict/atom_r_dict', 'rb')))
l_ICMF_dict = len(pickle.load(open('./common/dict/icmf_dict', 'rb')))
l_target_dict = len(pickle.load(open('./common/dict/target_dict', 'rb')))

parser = argparse.ArgumentParser()
parser.add_argument('--substructure', default='ICMF', help='Select the drug 2D substructure. (ICMF or ESPF)')
parser.add_argument('--device', default='cuda:0', help='Disables CUDA training.')
parser.add_argument('--batch-size', type=int, default=16, help='Number of batch_size')
parser.add_argument('--max-target', type=int, default=1000, help='The maximum length of the target.')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate. (human and celegans is 1e-4. Davis and KIBA is 5e-4)')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr-step-size', type=int, default=10, help='Period of learning rate decay.')
parser.add_argument('--lr-gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay.')
parser.add_argument('--count-decline', default=20, help='Early stoping.')
parser.add_argument('--task', default='DTI', help='Train DTI datasets.')
parser.add_argument('--drug-emb-dim', type=int, default=64, help='The embedding dimension of drug')
parser.add_argument('--target-emb-dim', type=int, default=64, help='The embedding dimension of target')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train.')
parser.add_argument('--p', type=int, default=10, help='The scale factor of coordinates.')
parser.add_argument('--K-FOLD', type=int, default=5, help='The 5 cross validation.')

args = parser.parse_args()

if args.device == 'cuda:0':
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


def train(model, optimizer, data_loader):
    model.train()
    pred_labels = []
    predictions = []
    all_labels = []
    total_loss = []
    for target, drug, labels in tqdm(data_loader):
        optimizer.zero_grad()

        out, loss2 = model(drug, target)

        labels = torch.FloatTensor(labels).to(args.device)
        loss = criterion(out.float(), labels.reshape(labels.shape[0]).long()) + loss2
        total_loss.append(loss.cpu().detach())
        ys = F.softmax(out, 1).to('cpu').data.numpy()
        pred_labels += list(map(lambda x: np.argmax(x), ys))  # (32)预测结果
        predictions += list(map(lambda x: x[1], ys))  # (32)标签为1的值
        all_labels += labels.cpu().numpy().reshape(-1).tolist()

        loss.backward()
        optimizer.step()
    res = evaluate_DTI(total_loss, pred_labels, predictions, all_labels)
    return res


def test(model, data_loader):
    model.eval()
    pred_labels = []
    predictions = []
    all_labels = []
    total_loss = []
    with torch.no_grad():
        for target, drug, labels in tqdm(data_loader, colour='blue'):

            out, loss2 = model(drug, target)

            labels = torch.FloatTensor(labels).to(args.device)
            loss = criterion(out.float(), labels.reshape(labels.shape[0]).long()) + loss2
            total_loss.append(loss.cpu().detach())
            ys = F.softmax(out, 1).to('cpu').data.numpy()
            pred_labels += list(map(lambda x: np.argmax(x), ys))  # (32)预测结果
            predictions += list(map(lambda x: x[1], ys))  # (32)标签为1的值
            all_labels += labels.cpu().numpy().reshape(-1).tolist()
    res = evaluate_DTI(total_loss, pred_labels, predictions, all_labels)
    return res


def run(model, optimizer, train_loader, valid_loader, test_loader, dataset, fold_i):
    os.makedirs('./save_model/{}/'.format(dataset), exist_ok=True)
    model_path = './save_model/model.pth'
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print(">>> Continue training ...")

    valid_max_AUC = 0.5
    decline = 0
    for epoch in range(1, args.epochs + 1):
        print('****** epoch:{} ******'.format(epoch))
        res_train = train(model, optimizer, train_loader)
        save_result_DTI(dataset, "Train", res_train, epoch)

        res_valid = test(model, valid_loader)
        save_result_DTI(dataset, "Valid", res_valid, epoch)

        res_test = test(model, test_loader)
        save_result_DTI(dataset, "Test", res_test, epoch)

        if res_valid[3] > valid_max_AUC:
            valid_max_AUC = res_valid[3]
            decline = 0
            save_path = './save_model/{}/{}_valid_best_checkpoint_{}_{}.pth'.format(dataset, args.substructure, fold_i, epoch)
            torch.save(model.state_dict(), save_path)
        else:
            decline = decline + 1
            if decline >= args.count_decline:
                print("EarlyStopping !!!")
                break

        if epoch % args.lr_step_size == 0:       # 每隔n次学习率衰减一次
            optimizer.param_groups[0]['lr'] *= args.lr_gamma


def save_data(model,test_loader):
    model_path = './save_model/human/ICMF_valid_best_checkpoint_4_39.pth'
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    res_test = test(model, test_loader)
    print("xxx")


if __name__ == '__main__':
    #############################################################
    # huamn and celegans: lr:1e-4, lr_step_size:10,  epochs: 40 #
    #############################################################

    datasets = ['human', 'celegans']
    common_data = load_common_data(args.substructure)
    if args.substructure.upper() == 'ICMF':
        l_substructures_dict = l_ICMF_dict
    else:
        l_substructures_dict = 23533

    for dataset in datasets:
        train_data = load_train_data_DTI(dataset)
        train_data_iter = MyDataset(common_data, train_data, args)

        for fold_i in range(args.K_FOLD):
            train_idx, test_idx = get_five_fold_idx(len(train_data_iter), fold_i, args.K_FOLD)
            train_dataset_iter = Subset(train_data_iter, train_idx)
            test_iter = Subset(train_data_iter, test_idx)

            valid_len = int(len(train_dataset_iter) * 0.1)
            train_len = len(train_dataset_iter) - valid_len
            train_iter, valid_iter = random_split(train_dataset_iter, [train_len, valid_len])

            my_collate_fn = partial(collate_fn, device=args.device)
            train_loader = DataLoader(train_iter, batch_size=args.batch_size, collate_fn=my_collate_fn, drop_last=True)
            valid_loader = DataLoader(valid_iter, batch_size=args.batch_size, collate_fn=my_collate_fn, drop_last=True)
            test_loader = DataLoader(test_iter, batch_size=args.batch_size, collate_fn=my_collate_fn, drop_last=True)

            model = MdDTI(l_drug_dict, l_target_dict, l_substructures_dict, args)
            model.to(args.device)

            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            run(model, optimizer, train_loader, valid_loader, test_loader, dataset, fold_i)
