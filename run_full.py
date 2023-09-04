from loguru import logger
import argparse
import torch
import time
import numpy as np
import random
time = 0
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(2021)
parser = argparse.ArgumentParser(description='REC')
parser.add_argument('--learning_rate', default='0.05', type=float,
                        help='Learning rate.')
parser.add_argument('--dropout_ration', default='0.2', type=float,
                        help='Drop out ration.')
parser.add_argument('--batch_size', default='1024', type=int,
                        help='batch size.')
parser.add_argument('--weight_negative', default='0.1', type=float,
                        help='weight for negative entry.')
parser.add_argument('--alpha', default='0.1', type=float,
                        help='alpha.')
parser.add_argument('--gpu', default='3', type=int,
                        help='Dataset name.')
parser.add_argument('--dataset', default='0', type=int,
                        help='0 for beibei and 1 for taobao.')
parser.add_argument('--v', default='0', type=int,
                        help='show the training process')
parser.add_argument('--weight_1', default='0.166666', type=float,
                        help='Dataset name.')
parser.add_argument('--weight_2', default='0.6666667', type=float,
                        help='Dataset name.')
parser.add_argument('--weight_3', default='0.166666', type=float,
                        help='Dataset name.')
args = parser.parse_args()
if args.dataset == 0:
    data_name = 'Beibei'
  
else:
    data_name = 'Taobao'

batch_size = args.batch_size

logger.info(args)

Epochs = 500


learning_rate = 0.05
dropout_ration = args.dropout_ration
dim_embedding = int(64)

lambdas = {'view':args.weight_1, 'cart':args.weight_2, 'buy':args.weight_3}

each_process_users = 64

torch.cuda.set_device(args.gpu)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

################################################################################################################################################################

import time
from itertools import chain
import pickle
with open('preprocess/'+data_name+'/view.pkl','rb') as load1:
    view = pickle.load(load1)
with open('preprocess/'+data_name+'/cart.pkl','rb') as load2:
    cart = pickle.load(load2)
with open('preprocess/'+data_name+'/buy_train.pkl','rb') as load3:
    buy_train = pickle.load(load3)
with open('preprocess/'+data_name+'/buy_test.pkl','rb') as load4:
    buy_test = pickle.load(load4)
num_whole_users = len(buy_train)
num_whole_items = max(list(chain.from_iterable(buy_train))+buy_test) + 1
max_length_test = max([num_whole_items-len(buy_train[i]) for i in range(num_whole_users)])
################################################################################################################################################################

import torch
from torch.utils.data import Dataset
class Read_Data(Dataset):
    def __init__(self, num_whole_users, num_whole_items, train_data_view=None, train_data_cart=None, train_data_buy=None):
        self.num_whole_users = num_whole_users
        self.num_whole_items = num_whole_items
        self.train_data_view = train_data_view
        self.train_data_cart = train_data_cart
        self.train_data_buy = train_data_buy

        self.max_train_length_view = max([len(self.train_data_view[i]) for i in range(self.num_whole_users)])
        self.max_train_length_cart = max([len(self.train_data_cart[i]) for i in range(self.num_whole_users)])
        self.max_train_length_buy = max([len(self.train_data_buy[i]) for i in range(self.num_whole_users)])
    def __getitem__(self, index):
        user_positive_items_view = self.train_data_view[index]
        user_positive_items_view.extend([self.num_whole_items]*(self.max_train_length_view-len(user_positive_items_view)))

        user_positive_items_cart = self.train_data_cart[index]
        user_positive_items_cart.extend([self.num_whole_items]*(self.max_train_length_cart-len(user_positive_items_cart)))

        user_positive_items_buy = self.train_data_buy[index]
        user_positive_items_buy.extend([self.num_whole_items]*(self.max_train_length_buy-len(user_positive_items_buy)))

        return index, torch.LongTensor(user_positive_items_view), torch.LongTensor(user_positive_items_cart), torch.LongTensor(user_positive_items_buy)
    def __len__(self):
        return self.num_whole_users
dataset = Read_Data(num_whole_users=num_whole_users, \
                    num_whole_items=num_whole_items, \
                    train_data_view=view, \
                    train_data_cart=cart, \
                    train_data_buy=buy_train)
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
################################################################################################################################################################
from model_full import Model
from utils import evaluation
time_list = []
import multiprocessing
if __name__ == '__main__':
    model = Model(num_users=num_whole_users, num_items=num_whole_items+1, dim_embedding=dim_embedding)
    model.cuda()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, initial_accumulator_value=1e-8)
    total_start = time.time()
    for epoch in range(Epochs):
        losses = 0 
        if time: 
            start = time.time() 
        for batch_data in dataloader:
            batch_users, batch_positive_items_view, batch_positive_items_cart, batch_positive_items_buy = batch_data
            model.forward(batch_users=batch_users.cuda(), whole_items=LongTensor(range(num_whole_items+1)), dropout_ration=dropout_ration)
            model.compute_positive_loss(batch_positive_items_view=batch_positive_items_view.cuda(), \
                                        batch_positive_items_cart=batch_positive_items_cart.cuda(), \
                                        batch_positive_items_buy=batch_positive_items_buy.cuda())
            batch_loss = model.compute_all_loss(weight_negative=args.weight_negative, lambdas=lambdas, alpha = args.alpha)
            losses = losses + batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        if args.v == 1:
            logger.info(' view_cart_buy : Epoch [{}/{}]'.format(epoch+1, Epochs))

    

        if epoch+1 in [10,200,500]:
            logger.info(' view_cart_buy : Epoch [{}/{}]'.format(epoch+1, Epochs))

            scores = []
            for step in range(0, int(num_whole_users/batch_size)+1):
                start = step * batch_size
                end = (step+1) * batch_size
                if end >= num_whole_users:
                    end = num_whole_users
                likelihood_buy = model.predict(batch_users=LongTensor(range(start,end)), whole_items=LongTensor(range(num_whole_items+1)))
                scores += likelihood_buy.cpu().tolist()

            pool = multiprocessing.Pool()
            results = []
            for step in range(0, int(num_whole_users/200)+1):
                start = step * 200
                end = (step+1) * 200
                if end >= num_whole_users:
                    end = num_whole_users
                result = pool.apply_async(evaluation, args=(200, num_whole_items, max_length_test, buy_train[start:end], scores[start:end], buy_test[start:end]))
                results.append(result)
            pool.close()
            pool.join()

            count_hr_10 = 0
            count_ndcg_10 = 0
            count_hr_50 = 0
            count_ndcg_50 = 0
            count_hr_100 = 0
            count_ndcg_100 = 0
            count_hr_200 = 0
            count_ndcg_200 = 0
            for result in results:
                x_10, y_10, x_50, y_50, x_100, y_100, x_200, y_200 = result.get()
                count_hr_10 += x_10
                count_ndcg_10 += y_10
                count_hr_50 += x_50
                count_ndcg_50 += y_50
                count_hr_100 += x_100
                count_ndcg_100 += y_100
                count_hr_200 += x_200
                count_ndcg_200 += y_200
            
            HR_10 = count_hr_10 / num_whole_users
            NDCG_10 = count_ndcg_10 / num_whole_users
            HR_50 = count_hr_50 / num_whole_users
            NDCG_50 = count_ndcg_50 / num_whole_users
            HR_100 = count_hr_100 / num_whole_users
            NDCG_100 = count_ndcg_100 / num_whole_users
            HR_200 = count_hr_200 / num_whole_users
            NDCG_200 = count_ndcg_200 / num_whole_users
            
            scores = []

            logger.info(' view_cart_buy | HR@10:{:.4f}, HR@50:{:.4f}, HR@100:{:.4f}, HR@200:{:.4f}'.format(HR_10, HR_50, HR_100, HR_200))
            logger.info(' view_cart_buy | NDCG@10:{:.4f}, NDCG@50:{:.4f}, NDCG@100:{:.4f}, NDCG@200:{:.4f}'.format(NDCG_10, NDCG_50, NDCG_100, NDCG_200))

