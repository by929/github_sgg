#! /usr/bin/python
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import random
import dill as pkl

from load_dataset import MyDataset, generate_filenames

class Net_Project(torch.nn.Module):
	def __init__(self, n_feat, n_label, n_pos, n_hidden, n_output):
		super(Net_Project, self).__init__()
		self.hidden_feat = torch.nn.Linear(n_feat, n_hidden)
		self.hidden_label = torch.nn.Linear(n_label, n_hidden)
		self.hidden_pos = torch.nn.Linear(n_pos, n_hidden)
		self.hidden = torch.nn.Linear(n_hidden * 3, 1024)
		self.predict = torch.nn.Linear(1024, n_output)
		# self.predict = torch.nn.Linear(n_hidden * 3, n_output)

	def forward(self, x):
		x_feat = self.hidden_feat(x[:, :4096])
		x_label = self.hidden_label(x[:, 4096:-4])
		x_pos = self.hidden_pos(x[:, -4:])
		x_final = torch.cat([x_feat, x_label, x_pos], 1)
		x_final = F.relu(x_final)
		x_final = F.relu(self.hidden(x_final))
		x_final = self.predict(x_final)
		return x_final

def load_data(mode, num_classes):
	print(("Load {} data").format(mode))
	filenames = generate_filenames(mode, 0)
	dataset = MyDataset(mode, filenames, num_classes)
	print(("Finish load {} data").format(mode))
	return filenames, dataset

if __name__ == '__main__':
	num_classes = 151
	epochs = 5
	lr = 0.001
	lr_step_size = 5000
	lr_decay_rate = 0.9
	batch_size = 128
	nraws = 1280
	modelPath = './baseline_model'

	net = Net_Project(4096, 151, 4, 1024, num_classes)
	optimizer = torch.optim.Adam(net.parameters(), lr = lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = lr_step_size, gamma = lr_decay_rate)

	# pos_weight = torch.ones([2])
	# loss_func = torch.nn.BCEWithLogitsLoss()
	loss_func = torch.nn.CrossEntropyLoss()

	use_gpu = torch.cuda.is_available()
	if use_gpu:
		net = net.cuda()
		loss_func = loss_func.cuda()

	for epoch in range(epochs):
		total = 0
		correct = 0
		net.train()

		train_filenames, trainset = load_data('train', num_classes)
		trainLoader = DataLoader(dataset = trainset, batch_size = batch_size, shuffle = True)

		for i, (_, x, y) in enumerate(trainLoader):
			x = Variable(x).float()
			y = Variable(y).type(torch.LongTensor)
			# y = Variable(y).float()

			if use_gpu:
				x = x.cuda()
				y = y.cuda()

			out = net(x)
			out = out.squeeze(dim=-1)
			
			_, gt_y = torch.max(y.data, 1)
			loss = loss_func(out, gt_y)
			# loss = loss_func(out, y)
			
			optimizer.zero_grad()  # clear gradients for next train
			loss.backward()        # backpropagation, compute gradients
			optimizer.step()
			scheduler.step()

			if (i+1) % 100 == 0:
				_, predicted = torch.max(out.data, 1)
				_, gt_y = torch.max(y.data, 1)
				correct += (predicted == gt_y).sum().item()
				total += y.size(0)

				print ('Epoch: %d/%d, Iter: %d,  Loss: %.4f, Accuracy: %.4f, Decaying learning rate to %g' 
					% (epoch + 1, epochs, i + 1, loss.item(), correct*100/total, scheduler.get_lr()[0]))

		eval_correct = 0
		eval_total = 0
		net.eval()

		test_filenames, testset = load_data('test', num_classes)
		testLoader = DataLoader(dataset = testset, batch_size = batch_size, shuffle = False)

		with torch.no_grad():
			test_img_id = np.ones(1)
			test_y_all = np.ones(1)
			test_pred_all = np.ones(1)
			for k, (img_id, test_x, test_y) in enumerate(testLoader):
				test_x = Variable(test_x).float()
				test_y = Variable(test_y).type(torch.LongTensor)
				# test_y = Variable(test_y).float()

				if use_gpu:
					test_x = test_x.cuda()
					test_y = test_y.cuda()
			
				test_out = net(test_x)
				test_out = test_out.squeeze(dim=-1)
				
				_, predicted = torch.max(test_out.data, 1)
				_, eval_gt_y = torch.max(test_y.data, 1)
				# predicted = test_out.data
				# eval_gt_y = test_y.data
				eval_correct += (predicted == eval_gt_y).sum().item()
				# eval_correct += (predicted == test_y).sum().item()
				eval_total += test_y.size(0)

				test_img_id = np.hstack([test_img_id, img_id.reshape([1,-1])[0]])
				test_y_all = np.hstack([test_y_all, eval_gt_y.cpu().data.numpy()])
				test_pred_all = np.hstack([test_pred_all, predicted.cpu().data.numpy()])

			print ('Correct: %d/%d, Accuracy: %.4f' \
				% (eval_correct, eval_total, eval_correct * 100 / eval_total))
			
			test_img_id = np.array(test_img_id[1:])
			test_y_all = np.array(test_y_all[1:])
			test_pred_all = np.array(test_pred_all[1:])

			fname = 'baseline_adam_'+ str(epoch + 1) +'.txt'
			f = open(os.path.join('baseline_result', fname), 'w')
			for box_id in range(test_img_id.shape[0]):
				f.write(str(test_img_id[box_id]) + ' ' + \
					str(test_y_all[box_id]) + ' ' + \
					str(test_pred_all[box_id]) + '\n')
			f.close()

		# save the model
		torch.save(net.state_dict(), os.path.join(modelPath, \
			'baseline_adam_lr0.001_' + str(epoch + 1) + '.pkl'))