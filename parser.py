from __future__ import absolute_import
import argparse

def arg_parse():
	parser = argparse.ArgumentParser()

	# datasets parameters
	parser.add_argument('--data_dir', type=str, default='data')
	parser.add_argument('--workers', default=4, type=int, help="number of data loading workers (default: 4)")
	
	# training parameters
	parser.add_argument('--gpu', default=0, type=int)
	parser.add_argument('--n_epoch', default=300, type=int)
	parser.add_argument('--warm_up_epoch', default=10, type=int)
	parser.add_argument('--train_batch', default=64, type=int)
	parser.add_argument('--lr', default=5e-3, type=float)
	parser.add_argument('--lr_min', default=5e-5, type=float)
	parser.add_argument('--weight_decay', default=0.0001, type=float)
	parser.add_argument('--momentum', default=0.0001, type=float)

	parser.add_argument('--n_landmark', default=68, type=int)
	parser.add_argument('--num_model', default=2, type=int)
	
	# others
	parser.add_argument('--save_dir', default='log', type=str)
	parser.add_argument('--save_model_name', default='model_best.pth', type=str)
	parser.add_argument('--random_seed', default=999, type=int)
	parser.add_argument('--num_pic', default=50, type=int, help='number of pictures to be generated for visualization')


	args = parser.parse_args()

	return args