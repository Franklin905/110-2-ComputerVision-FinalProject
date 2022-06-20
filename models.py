import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
	def __init__(self, args):
		super(BaseModel, self).__init__()
		self.net = nn.Sequential(
						nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
						nn.BatchNorm2d(64),
						nn.ReLU(True),
						nn.MaxPool2d(2, 2, 0),

						nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
						nn.BatchNorm2d(128),
						nn.ReLU(True),
						nn.MaxPool2d(2, 2, 0),

						nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
						nn.BatchNorm2d(256),
						nn.ReLU(True),
						nn.MaxPool2d(2, 2, 0),

						nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
						nn.BatchNorm2d(384),
						nn.ReLU(True),
						nn.MaxPool2d(4, 4, 0),
					)

		self.fc = nn.Sequential(
					nn.Linear(384*1*1, 256),
					nn.GELU(),
					nn.Linear(256, args.n_landmark*2),
				)

	def forward(self, img):
		out = self.net(img)
		feats = out.view(out.size(0), -1)
		out = self.fc(feats)

		return out, feats
