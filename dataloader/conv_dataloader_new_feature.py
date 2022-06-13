from __future__ import print_function, division
import os
import torch 
import numpy as np 
import csv
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import re
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
feature_index = {}
feature_idx = 0
class conv_seq_dataset_new_feature(Dataset):
	"sensor and pressure dataset"

	def __init__(self, csv_file,transform = None,pick = False,seq = False,div = None, train = True, split_ratio = 0.8, quick = False):

		self.raw_data = pd.read_csv(csv_file)
		self.transform = transform

		if(quick == True):
			if(train == True):
				with open("dataloader/single_data_train", 'rb') as f:
					self.combi_data = pickle.load(f)
			else:
				with open("dataloader/single_data_test.txt", 'rb') as f:
					self.combi_data = pickle.load(f)

		else:
			print("start grouping")
			if(pick == False):
				# self.combi_data = self.group_press()
				print("first grouping done")
				# self.max_length = self.get_max()
				self.max_length = 60
				self.combi_data = self.group_press(True)
				with open(csv_file.split('.')[0] + '_conv_'+str(div)+'.pickle','wb') as handle:
					pickle.dump(self.combi_data, handle, protocol = pickle.HIGHEST_PROTOCOL)
			else:
				with open(csv_file.split('.')[0] + '_conv_'+str(div)+'.pickle','rb') as handle:
					self.combi_data = pickle.load(handle)
			if(div != None):
				self.divide_data(div)

			length = int(self.combi_data['sensor'].shape[0]*(split_ratio/2))


			if(train == True):
				range_slice = []
				for s in range(0, length):
					range_slice.append(s)
				for s in range(-length, 0):
					range_slice.append(s)
				self.combi_data['sensor'] = self.combi_data['sensor'][range_slice]
				self.combi_data['pressure'] = self.combi_data['pressure'][range_slice]
				self.combi_data['coord'] = self.combi_data['coord'][range_slice]
				self.combi_data['map'] = self.combi_data['map'][range_slice]

			else:
				self.combi_data['sensor'] = self.combi_data['sensor'][length:-length]
				self.combi_data['pressure'] = self.combi_data['pressure'][length:-length]
				self.combi_data['coord'] = self.combi_data['coord'][length:-length]
				self.combi_data['map'] = self.combi_data['map'][length:-length]



		print("grouping done")


	def __len__(self):
		return len(self.combi_data['sensor'])
	def divide_data(self, window_size):
		total_data = {'sensor' : [], 'pressure' : [], 'coord' : [], 'map':[]}
		for i in range(len(self.combi_data['sensor'])):
			for j in range(len(self.combi_data['sensor'][0]) - window_size +1):
				if(len(self.combi_data['sensor'][0]) - window_size +1 != 53):
					print("each section length is: ",len(self.combi_data['sensor'][0]) - window_size +1)

				total_data['sensor'].append(self.combi_data['sensor'][i][j:j+window_size][:])
				total_data['pressure'].append(self.combi_data['pressure'][i][j:j+window_size])
				total_data['coord'].append(self.combi_data['coord'][i][j:j+window_size])
				total_data['map'].append(self.combi_data['map'][i][j:j+window_size])

		total_data['coord'] = np.array(total_data['coord'])
		total_data['map'] = np.array(total_data['map'])

		total_data['sensor'] = np.array(total_data['sensor'])
		total_data['pressure'] = np.array(total_data['pressure'])

		self.combi_data = total_data
		# print(self.combi_data['sensor'].shape)
	def coord_from_pos(self,pos):

		tmp = pos.split("'")
		
		return tmp[1], tmp[3] 

	def heatmap(self,coord, pressure):
		stride = 5.75/2
		# x, y = np.mgrid[-23:23.1:stride, -23:23.1:stride]
		x, y = np.mgrid[-28.75:28.75:stride, -28.75:28.75:stride]
		xy = np.column_stack([x.flat, y.flat])
		xcoord, ycoord = coord[0], coord[1]
		mu = np.array([xcoord, ycoord])
		sigma = np.array([5.75, 5.75])
		covariance = np.diag(sigma ** 2)
		z = multivariate_normal.pdf(xy, mean=mu, cov=covariance,)
		alpha = pressure / np.max(z)
		z = z * alpha
		z = z.reshape(x.shape)
		return z


	def get_row(self, idx):
		idx_data = self.raw_data.iloc[idx,:]
		sensor_data = np.array(idx_data[:25]).astype(float)
		# for i in range(5):
		# 	idx = 5*i
		# 	sensor_data[5*i:5*i+5] = sensor_data[[idx, idx+1, idx+3,idx+4,idx+2]]
		sensor_data = np.array(idx_data[:25]).astype(float).reshape((5,5))
		pressure_data = idx_data[25]
		position = idx_data[26]
		x, y = self.coord_from_pos(position)

		tf = idx_data[27]
		coord = np.array([x,y]).astype(float)
		z = self.heatmap(coord, pressure_data)
		item = {'sensor':sensor_data,'pressure' : pressure_data, 'coord' :coord, 'map' :z, 'tf' : tf}
		# if(self.transform):
		# 	item = self.transform(item)
		return item
	def get_max(self):
		max = 0
		for s in self.combi_data['pressure']:
			if(s.shape[0]>max):
				max = s.shape[0]
		print("max length is: ",max)
		return max


	def group_press(self,padding = False):
		i = 0
		item = self.get_row(i)

		pre_state = False
		total_data = {'sensor' : [], 'pressure' : [], 'coord' : [],'map':[]}

		sensor_data = None
		while(i<len(self.raw_data)):
			if(i%10 == 0):
				print(i)
			item = self.get_row(i)
			if(-10>item['pressure'] or 1000<item['pressure']):
				i+=1
				continue
			cur_state = item['tf']
			if(cur_state == True and pre_state == False):
				p_sensor = np.array([item['sensor']])
				p_pressure = np.array([item['pressure']])
				p_coord = np.array([item['coord']])
				p_map = np.array([item['map']])
				pre_state = cur_state
			elif(cur_state == False and pre_state == True):
				if(padding == True):
					num2add = self.max_length - len(p_sensor)
					for j in range(1,num2add+1):
						tmp_item = self.get_row(i+j)
						p_sensor = np.append(p_sensor,[tmp_item['sensor']],axis = 0)
						p_pressure = np.append(p_pressure,[tmp_item['pressure']],axis = 0)
						p_coord = np.append(p_coord,[tmp_item['coord']],axis = 0)
						p_map = np.append(p_map,[tmp_item['map']],axis = 0)

				pre_state = cur_state
				total_data['sensor'].append(p_sensor)
				total_data['pressure'].append(p_pressure)
				total_data['coord'].append(p_coord)
				total_data['map'].append(p_map)

			elif(cur_state == True and pre_state == True):
				p_sensor = np.append(p_sensor,[item['sensor']],axis = 0)
				p_pressure = np.append(p_pressure,[item['pressure']],axis = 0)
				p_coord = np.append(p_coord,[item['coord']],axis = 0)
				p_map = np.append(p_map,[item['map']],axis = 0)
			i+=1
		total_data['sensor'] = np.array(total_data['sensor'])
		total_data['pressure'] = np.array(total_data['pressure'])
		total_data['coord'] = np.array(total_data['coord'])
		total_data['map'] = np.array(total_data['map'])
		return total_data


	def __getitem__(self,idx):
		item = {'sensor': self.combi_data['sensor'][idx],'pressure' : self.combi_data['pressure'][idx],'coord' : self.combi_data['coord'][idx],'map':self.combi_data['map'][idx]}
		if(self.transform):
			item = self.transform(item)
		return item


class ToTensor(object):
	def __call__(self, item):
		sensor, pressure, coord, heatmap = item['sensor'], item['pressure'],item['coord'],item['map']
		sensor, pressure, coord, heatmap = torch.from_numpy(sensor),torch.from_numpy(pressure), torch.from_numpy(coord), torch.from_numpy(heatmap)
		return {"sensor" : sensor, "pressure" : pressure, "coord" : coord,'map':heatmap}




if __name__ == "__main__":
	pass
