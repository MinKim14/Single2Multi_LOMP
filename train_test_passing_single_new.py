import matplotlib
# matplotlib.use('Agg')
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import matplotlib.pyplot as plt 
import pandas as pd
#import seaborn as sn 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sys, os
from os.path import dirname, join, abspath
sys.path.insert(0, './dataloader')
from conv_dataloader_new_feature  import conv_seq_dataset_new_feature, ToTensor
from sklearn.model_selection import train_test_split
import matplotlib as mpl

from modi_net import passing_Net
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# from touch2_dataset import touch2_dataset_generation
# multi_dataset = touch2_dataset_generation(dataset_h, stride = 5.75/2,transform = transforms.Compose([ToTensor()]), train = True)
# multi_dataloader = DataLoader(multi_dataset, batch_size = 500, shuffle = False)

train_dataset =conv_seq_dataset_new_feature('dataloader/data/dense_5x5_1.csv', transform = transforms.Compose([ToTensor()]),pick = True, seq = True, div = 8,train = True,split_ratio = 0.8)
test_dataset = conv_seq_dataset_new_feature('dataloader/data/dense_5x5_1.csv', transform = transforms.Compose([ToTensor()]),pick = True, seq = True, div = 8,train = False,split_ratio = 0.8)
train_dataloader = DataLoader(train_dataset,batch_size = 1000, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = 500, shuffle = False)


best_eval_loss = 1e99
best_eval_epoch = 0
cor_test_loss = 1e99
cor_test_epoch = 0

hidden_size = 128
multiple = 1
criterion = nn.MSELoss()
model = passing_Net(1, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train(train_dataloader, model, epoch):
	global multiple

	total_loss = 0
	total_size = 0
	total_feature_loss = 0
	for i, s in enumerate(train_dataloader):
		sen_val, target_map = s['sensor'].float().to(device), s['map'].float().to(device)
		model.train()
		target_map = target_map[:,-1]
		hz = torch.randn(sen_val.size(0), 25, hidden_size).to(device)
		for j in range(8):
			jth_sen_val = sen_val[:,j]
			heatmap, hz = model(jth_sen_val, hz)
		loss = criterion(heatmap, target_map)
		optimizer.zero_grad()
		loss.backward()
		for param in model.parameters():
			param.grad.data.clamp_(-1,1)
		optimizer.step()
		total_loss += loss.item() * sen_val.size(0)
		total_size +=sen_val.size(0)
	

	print(total_loss/total_size)
	print("epoch is: ",epoch, "training_loss: ",total_loss/total_size, " feature loss is: ",total_feature_loss/total_size)


def multi_train(train_dataloader, model, epoch):
	global multiple

	total_loss = 0
	total_size = 0
	total_feature_loss = 0
	for i, s in enumerate(train_dataloader):
		sen_val, target_map = s['sen'].float().to(device), s['map'].float().to(device)
		model.train()
		hz = torch.randn(sen_val.size(0), 25, hidden_size).to(device)
		for j in range(8):
			jth_sen_val = sen_val[:,j]
			heatmap, hz = model(jth_sen_val, hz)
		loss = criterion(heatmap, target_map)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		total_loss += loss.item() * sen_val.size(0)
		total_size +=sen_val.size(0)

	print(total_loss/total_size)
	print("epoch is: ",epoch, "training_loss: ",total_loss/total_size, " feature loss is: ",total_feature_loss/total_size)
 

test_best_epoch = 0
def test(test_dataloader, model, epoch):
	global best_eval_loss, A, test_best_epoch
	total_loss = 0
	total_size = 0
	for i, s in enumerate(test_dataloader):

		sen_val, target_map = s['sensor'].float().to(device), s['map'].float().to(device)
		model.eval()
		target_map = target_map[:,-1]
		hz = torch.randn(sen_val.size(0), 25, hidden_size).to(device)
		for j in range(8):
			jth_sen_val = sen_val[:,j]
			heatmap, hz = model(jth_sen_val, hz)
		loss = criterion(heatmap, target_map)
		total_loss += loss.item() * sen_val.size(0)
		total_size +=sen_val.size(0)
	test_loss = total_loss/total_size
	if(test_loss < best_eval_loss):
		torch.save(model, "./model_log/model_"+str(test_loss)+".pkl")
		best_eval_loss = test_loss
		test_best_epoch = epoch
	print("testing loss is: ", total_loss/total_size," best_loss: ", best_eval_loss, " test best epoch: ",test_best_epoch)


multi_best_loss = 1e99
multi_best_epoch = 0
def multi_test(test_dataloader, model, epoch):
	global multi_best_loss,multi_best_epoch, A 
	total_loss = 0
	total_size = 0
	for i, s in enumerate(test_dataloader):
		sen_val, target_map = s['sen'].float().to(device), s['map'].float().to(device)
		model.eval()
		hz = torch.randn(sen_val.size(0), 25, hidden_size).to(device)
		for j in range(8):
			jth_sen_val = sen_val[:,j]
			heatmap, hz = model(jth_sen_val, hz)
		loss = criterion(heatmap, target_map)
		total_loss += loss.item() * sen_val.size(0)
		total_size +=sen_val.size(0)
	test_loss = total_loss/total_size
	if(test_loss < multi_best_loss):
		multi_best_loss = test_loss
		multi_best_epoch = epoch
	print("Multi loss is: ", total_loss/total_size," best_loss: ", multi_best_loss, "best_epoch: ",multi_best_epoch	)



if __name__ == "__main__":

	# from distance_based_dataset import pickle_load_dataset
	# dist_multi_dataset1 = pickle_load_dataset(8)
	# dist_multi_dataloader = DataLoader(dist_multi_dataset1, batch_size = 1000, shuffle = False)
	for i in range(5000):
		train(train_dataloader, model,i)
		test(test_dataloader, model, i)
		# if(i%5 ==1):
		# 	multi_test(multi_dataloader, model, i)


