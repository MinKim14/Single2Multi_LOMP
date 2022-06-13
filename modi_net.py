import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import optim

import sys, os
from os.path import dirname, join, abspath
import random
import time

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class passing_Net(nn.Module):
	def __init__(self, input_size, hidden_size, num_cell=25):
		super(passing_Net, self).__init__()
		"""
		input_size: size of input vector
		hidden_size: size of hidden vector generated by GRU cells
		num_cell: number of sensor cells in matrix sensor e.g. 5x5 => 25
		"""
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.LSTM_cells = []
		self.num_cell = num_cell


		self.num_layers = 3
		self.dropout = nn.Dropout(0.3)

		self.path_relu = nn.ReLU()
		self.relu = nn.LeakyReLU()
		self.m_relu = nn.LeakyReLU()
		for i in range(25):
			self.add_module('i2h_'+str(i), nn.GRUCell(self.input_size, self.hidden_size).to(self.device))
			self.add_module('linear_'+str(i) , nn.Sequential(
				nn.Linear(self.hidden_size, self.hidden_size),
				nn.LeakyReLU(),
				nn.Linear(self.hidden_size, 16),
			))
			self.add_module('path_' + str(i), nn.Sequential(
				nn.Linear(self.hidden_size + 1, self.hidden_size),
				nn.LeakyReLU(),
				nn.Linear(self.hidden_size, self.hidden_size +1),
			))
			self.add_module('last_path_' + str(i), nn.Sequential(
				nn.Linear(self.hidden_size*2 + 2, self.hidden_size),
				nn.LeakyReLU(),
				nn.Linear(self.hidden_size, self.hidden_size),
				nn.LeakyReLU(),
			))

		self.i2h = AttrProxy(self, 'i2h_')
		self.all_linear = AttrProxy(self, 'linear_')
		self.all_path = AttrProxy(self,'path_')
		self.last_all_path = AttrProxy(self, 'last_path_')

		self.gen_passage()




		
	def gen_passage(self):
		self.all_path_dict = {}
		for i in range(25):
			self.all_path_dict[i] = []

		for i in range(5):
			for j in range(5):
				cur_ver_idx = i * 5 + j
				for x in range(5):
					for y in range(5):
						dist = abs(i-x) + abs(j-y)
						to_ver_idx = x * 5 + y
						if(dist == 1):
							self.all_path_dict[cur_ver_idx].append(to_ver_idx)
						elif(dist == 2 and (i!= x) and (j!=y)):
							self.all_path_dict[cur_ver_idx].append(to_ver_idx)
		print("self.all_path is: ",self.all_path_dict)

	def forward(self, inputs, prev_hidden):
		"""
			lets regard input shape as (N, 25, 1)
		"""
		batch_size = inputs.size(0)
		inputs = inputs.reshape(batch_size, -1)

		new_hz = []
		for i, i2h in zip(range(self.num_cell), self.i2h):
			ith_cell_input = inputs[:, i].reshape(-1, 1)
			ith_prev_hz = prev_hidden[:, i]
			out_hz = i2h(ith_cell_input, ith_prev_hz)
			out_hz = self.relu(out_hz)
			out_hz = torch.cat((out_hz, ith_cell_input),1)
			new_hz.append(out_hz)

		updated_hz = []
		linear_out_indiv = []
		for i, path, last_path, i_linear in zip(range(self.num_cell), self.all_path, self.last_all_path, self.all_linear):
			hz_vec = new_hz[i].clone()
			cnt = 0
			for s in self.all_path_dict[i]:
				from_idx = s
				cur_vec = 0.2 * path(new_hz[from_idx].clone())
				hz_vec += cur_vec

			con_cur_vec = torch.cat((new_hz[i],hz_vec),1)
			cur_vec = last_path(con_cur_vec)
			updated_hz.append(cur_vec)
			hidden_out = i_linear(cur_vec).reshape(-1,1,4,4)
			linear_out_indiv.append(hidden_out)


		linear_out = torch.cat(linear_out_indiv, 1)

		y_linear_out = []
		for i in range(4,-1,-1):
			cur_idx = i * 5
			yith_linear_out = torch.cat((linear_out[:,cur_idx+4],linear_out[:,cur_idx +3],linear_out[:,cur_idx +2],linear_out[:,cur_idx +1],linear_out[:,cur_idx]),2)
			y_linear_out.append(yith_linear_out)
		cat_linear_out = torch.cat(y_linear_out, 1)
		return cat_linear_out, torch.cat(updated_hz, 1).reshape(batch_size, self.num_cell, -1),  linear_out_indiv



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
	test_net = passing_Net(1, 64).to(device)
	


