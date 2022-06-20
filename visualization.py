import os
import numpy as np

import torch
import torch.nn as nn
import torchvision

import parser
import utility
import models
import ConvNext
import data


if __name__=='__main__':

	args = parser.arg_parse()

	torch.set_default_dtype(torch.float32)
	torch.set_default_tensor_type(torch.FloatTensor)

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	
	''' setup random seed '''
	myseed = args.random_seed
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(myseed)
	torch.manual_seed(myseed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(myseed)

	device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

	print('===> prepare dataloader ...')
	train_loader = torch.utils.data.DataLoader(data.DATA_vis(args.data_dir, mode='train'),
											   batch_size=args.train_batch, 
											   num_workers=args.workers,
											   shuffle=True,
											   pin_memory=True)
	val_loader   = torch.utils.data.DataLoader(data.DATA_vis(args.data_dir, mode='val'),
											   batch_size=args.train_batch,
											   num_workers=args.workers,
											   shuffle=False,
											   pin_memory=True)
	test_loader  = torch.utils.data.DataLoader(data.DATA_vis(args.data_dir, mode='test'),
											   batch_size=args.train_batch,
											   num_workers=args.workers,
											   shuffle=False,
											   pin_memory=True)

	print('===> prepare model ...')
	model = ConvNext.ReducedConvNeXt(in_chans=3, num_classes=args.n_landmark*2, depths=[3, 9, 3], dims=[96, 192, 384], drop_path_rate=0.1)

	model.load_state_dict(torch.load(args.save_model_name, map_location='cpu'))
	model.to(device)

	print('===> start visualizing ...')
	model.eval()

	with torch.no_grad():
		for idx, (img_names, imgs, gts) in enumerate(train_loader):
			imgs = imgs.to(device); gts = gts.float().to(device)
			batch_size = imgs.size(0)

			preds, _ = model(imgs)
			preds = preds.view(batch_size, -1, 2).cpu().numpy()
			flag = False
			for i in range(batch_size):
				img_path = os.path.join(args.data_dir, 'synthetics_train', img_names[i])
				save_path = os.path.join(args.save_dir, 'train_' + img_names[i])
				utility.plot_coordinates(img_path, save_path, preds[i], gts[i].cpu().numpy())

				if idx*args.train_batch + i + 1 == args.num_pic:
					flag = True
					break
			if flag:
				break


		for idx, (img_names, imgs, gts) in enumerate(val_loader):
			imgs = imgs.to(device); gts = gts.float().to(device)
			batch_size = imgs.size(0)

			preds, _ = model(imgs)
			preds = preds.view(batch_size, -1, 2).cpu().numpy()
			flag = False
			for i in range(batch_size):
				img_path = os.path.join(args.data_dir, 'aflw_val', img_names[i])
				save_path = os.path.join(args.save_dir, 'val_' + img_names[i])
				utility.plot_coordinates(img_path, save_path, preds[i], gts[i].cpu().numpy())

				if idx*args.train_batch + i + 1 == args.num_pic:
					flag = True
					break
			if flag:
				break

		for idx, (img_names, imgs, _) in enumerate(test_loader):
			imgs = imgs.to(device)
			batch_size = imgs.size(0)

			preds, _ = model(imgs)
			preds = preds.view(batch_size, -1, 2).cpu().numpy()
			flag = False
			for i in range(batch_size):
				img_path = os.path.join(args.data_dir, 'aflw_test', img_names[i])
				save_path = os.path.join(args.save_dir, 'test_' + img_names[i])
				utility.plot_coordinates(img_path, save_path, preds[i], None)

				if idx*args.train_batch + i + 1 == args.num_pic:
					flag = True
					break
			if flag:
				break
