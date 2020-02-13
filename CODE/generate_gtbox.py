import numpy as np
import dill as pkl
import os
from collections import defaultdict


img_boxes = defaultdict(list)
img_labels = defaultdict(list)


def generate_filenames(mode, start_i):
	filenames = []
	if mode == 'test':
		for i in range(start_i, 26000, 1000):
			filename = 'vg_{}_{}-{}.txt'.format(mode, i, i+1000)
			filenames.append(filename)
		filenames.append('vg_test_26000-26446.txt')
	elif mode == 'train':
		for i in range(start_i, 57000, 1000):
			filename = 'vg_{}_{}-{}.txt'.format(mode, i, i+1000)
			filenames.append(filename)
		filenames.append('vg_train_57000-57723.txt')
		# np.random.shuffle(filenames)
	return filenames

def generate_box(f):
	global img_boxes, img_labels
	datum = f.readlines()
	for line in datum:
		data = line.strip().split(' ')
		data = np.asarray(data, dtype = np.float)
		img_labels[data[0]].append(data[5])
		img_boxes[data[0]].append(data[-6:-2])

def read_files():
	mode = 'test'
	path = '{}/{}_sgcls_txt'.format(mode, mode)
	filenames = generate_filenames(mode, 0)
	for filename in filenames:
		print(filename)
		f = open(os.path.join(path, filename), 'r')
		generate_box(f)
		f.close()


def read_txt(fpred_path):
	'''
	fgt文件保存内容：img_id box_id box_label x1 x2 y1 y2
	pred文件保存内容: img_id box_gt_label box_pred_label
	'''
	fgt = open('test/test_gt_box.txt', 'r')
	datum = fgt.readlines()
	fpred = open(fpred_path, 'r')
	datum_pred = fpred.readlines()
	for i in range(len(datum)):
		data = datum[i].strip().split(' ')
		data = np.asarray(data, dtype = np.float)
		pred = datum_pred[i].strip().split(' ')
		pred = np.asarray(pred, dtype = np.float)
		assert data[0] == pred[0] and data[2] == pred[1]
		img_labels[data[0]].append(pred[2])
		img_boxes[data[0]].append(data[-4:])
	fgt.close()
	fpred.close()


if __name__ == '__main__':
	img_num = 26446
	# read_files()

	save_pkl = 'baseline_sgcls_modify_label.pkl'
	fpred_path = 'baseline_result/baseline_5.txt'
	read_txt(fpred_path)

	all_imgs = defaultdict(dict)
	for i in range(img_num):
		all_imgs[i]['boxes'] = np.asarray(img_boxes[i], dtype = np.float)
		all_imgs[i]['labels'] = np.asarray(img_labels[i], dtype = np.float)
	with open(save_pkl,'wb') as fpkl:
		pkl.dump(all_imgs, fpkl)