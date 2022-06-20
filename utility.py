import os
import cv2
import math
import pickle
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

class SmoothWingLoss(nn.Module):
	def __init__(self, t=0.01, w=10, eps=2):
		super(SmoothWingLoss, self).__init__()
		"""
		Implementation of Smooth-Wing Loss introduced by LOTR (https://arxiv.org/abs/2109.10057) (IEEE Access)
		# t: inner threshold
		# w: outer threshold
		# eps: steepness control parameter
		"""

		self.t = t
		self.w = w
		self.eps = eps
		self.s = (w + eps) / (2 * t * (eps + t))
		self.c1 = w - (w + eps) * math.log(1 + w / eps)
		self.c2 = self.s * t * t

	def forward(self, preds, gts):
		# preds size = (B, num of landmarks, 2)
		assert preds.shape == gts.shape, "preds shape = {} != gts.shape = {}".format(preds.shape, gts.shape)

		diff = torch.abs(gts - preds)

		first_condition = (diff < self.t).type(torch.float32)
		second_condition = (diff > self.w).type(torch.float32)
		third_condition = torch.logical_not(torch.logical_or(first_condition, second_condition)).type(torch.float32)

		first_loss = self.s * torch.square(diff)
		second_loss = diff - self.c1 - self.c2
		third_loss = ((self.w + self.eps) * torch.log(1 + diff / self.eps)) - self.c2

		# loss for each coordinate
		loss_elements = torch.sum(torch.stack([torch.mul(first_loss, first_condition), torch.mul(second_loss, second_condition),\
									torch.mul(third_loss, third_condition)]), dim=0)	# size = (B, num of landmarks, 2)
		total_loss = torch.mean(torch.sum(loss_elements, (2, 1)))	# sum all the loss of a face, and average over the batch size
		return total_loss

def calculate_NME(preds, gts):
	# Calculate normalized mean error (NME)
	# preds and gts should be same type (numpy array or tensor) and same size
	# preds size = (B, num of landmarks, 2)
	assert preds.shape == gts.shape, "preds shape = {} != gts.shape = {}".format(preds.shape, gts.shape)

	diff = preds - gts
	if torch.is_tensor(preds):
		NME = torch.sum(torch.sqrt(torch.sum(torch.square(diff), 2)), 1) / 384
	else:
		NME = np.sum(np.sqrt(np.sum(np.power(diff, 2), 2)), 1) / 384

	return NME

def evaluate(model, data_loader, device):
	model.eval()

	nme_list = []
	with torch.no_grad():
		for (imgs, gts) in data_loader:
			imgs = imgs.to(device); gts = gts.float().to(device)
			batch_size = imgs.size(0)

			preds, feats = model(imgs)
			preds = preds.view(batch_size, -1, 2).contiguous()
			nme_list.append(calculate_NME(preds, gts))

		mean_nme = torch.mean(torch.cat(nme_list, dim=0))

		return mean_nme

def generate_test_txt(model, data_loader, device):
	model.eval()

	f = open('solution.txt', 'w')
	with torch.no_grad():
		for (imgs, img_names) in data_loader:
			imgs = imgs.to(device)
			batch_size = imgs.size(0)

			preds, _ = model(imgs)
			preds = preds.detach().cpu().numpy()
			for i in range(batch_size):
				f.write(img_names[i] + ' ')
				for j in range(preds.shape[1]):
					f.write('{:.4f}'.format(preds[i][j]))
					if j != preds.shape[1] - 1:
						f.write(' ')
					else:
						f.write('\n')
	f.close()

def read_gt_coordinates(pkl_data_path):
	with open(pkl_data_path, 'rb') as f:
		X, Y = pickle.load(f)
		return X, Y

def plot_coordinates(img_path, save_path, coords_1, coords_2=None):
	# Read image from the given path
	# Could plot ground truth coordinates or predicted coordinates
	
	img = cv2.imread(img_path, cv2.IMREAD_COLOR)
	for x, y in coords_1:
		x = x.astype(np.int32); y = y.astype(np.int32)
		cv2.circle(img, (x, y), 2, (0, 0, 255), -1)		# color code (B, G, R)

	if isinstance(coords_2, np.ndarray):
		for x, y in coords_2:
			x = x.astype(np.int32); y = y.astype(np.int32)
			cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

	cv2.imwrite(save_path, img)
