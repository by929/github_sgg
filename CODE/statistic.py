import numpy as np
import os

box_id = 0
box_faster_cnt = 0
box_motif_cnt = 0

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

# 统计faster-rcnn和motif预测正确的box数量
def count_label(f):
	global box_id, box_faster_cnt, box_motif_cnt
	datum = f.readlines()
	for line in datum:
		data = line.strip().split(' ')
		if float(data[2]) == float(data[3]):
			box_faster_cnt += 1
		if float(data[2]) == float(data[5]):
			box_motif_cnt += 1
		box_id += 1


if __name__=="__main__":
	mode = 'test'
	filenames = generate_filenames(mode, 0)
	path = '{}/{}_sgcls_txt'.format(mode, mode)
	for filename in filenames:
		file = open(os.path.join(path, filename), 'r')
		count_label(file)
		file.close()
		print(filename, box_id, box_faster_cnt, box_motif_cnt)