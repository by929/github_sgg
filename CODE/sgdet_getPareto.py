import numpy as np
import dill as pkl
import os
from collections import defaultdict
import itertools
from itertools import combinations, product
import time

'''
此文件用于求测试集的predbox的pareto最优解；
最优解保存于，
是一个defaultdict(list)，key=图片id，
value=pareto最优解，即n维向量(w1,...,wn)，
'''

img_id = 0

# 获取训练集中任意两个label之间的iou和labelset的overlap
label_iou = pkl.load(open('train/sgdet_train/train_label_iou_noconstraint.pkl', 'rb'))
labelset_overlap = pkl.load(open('train/sgdet_train/train_labelset_overlap_noconstraint.pkl', 'rb'))
print("finish load pkl")

sum_d = 0
for k in labelset_overlap:
	sum_d += labelset_overlap[k]['var_overlap']
mean_d = sum_d / len(labelset_overlap)	# 平均方差mean_d=0.001996


# 对图像中像素点进行分类
def get_pixeltype(img_predbox):
	box_scope = np.array(img_predbox[:, -4:], dtype=np.int32)
	num_boxes = len(box_scope)
	pixel = defaultdict(dict)
	
	for i in range(1026):
		for j in range(1026):
			pixel[i][j] = set()

	for i in range(num_boxes):
		x1, y1, x2, y2 = box_scope[i]
		for x in range(x1, x2+1):
			for y in range(y1, y2+1):
				pixel[x][y].add(i)

	pixeltype = dict()
	for i in range(1026):
		for j in range(1026):
			if len(pixel[i][j]) == 0:
				continue
			elif tuple(pixel[i][j]) in pixeltype:
				pixeltype[tuple(pixel[i][j])] += 1
			else:
				pixeltype[tuple(pixel[i][j])] = 1
	print("len(pixeltype): ", len(pixeltype))
	return pixeltype


# 计算boxes的面积并集
def get_box_overlap(boxes):
	area = np.zeros([1026, 1026])	# x2和y2的最大值为1024
	for box in boxes:
		x1, y1, x2, y2 = np.array(box, dtype=np.int32)
		area[x1:x2+1, y1:y2+1] = 1
	return np.sum(area)


def get_pixel_overlap(wset, pixeltype):
	sum_pixel = 0
	for k in pixeltype:
		if len(set(wset) & set(k)) != 0:
			sum_pixel += pixeltype[k]
	return sum_pixel


# 计算两个box之间的iou
def get_iou(box1, box2):
	xmin1, ymin1, xmax1, ymax1 = np.array(box1, dtype=np.int32)
	xmin2, ymin2, xmax2, ymax2 = np.array(box2, dtype=np.int32)
	s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
	s2 = (xmax2 - xmin2) * (ymax2 - ymin2)
	xmin_i = max(xmin1, xmin2)
	xmax_i = min(xmax1, xmax2)
	ymin_i = max(ymin1, ymin2)
	ymax_i = min(ymax1, ymax2)
	if xmin_i >= xmax_i or ymin_i >= ymax_i:
		return 0
	s_i = (xmax_i - xmin_i) * (ymax_i - ymin_i)
	s_u = s1 + s2 - s_i
	return float(s_i) / float(s_u)


# 计算图像中任意两个box之间的iou
def get_img_iou(img_predbox):
	img_iou = defaultdict(dict)
	box_num = len(img_predbox)
	for i in range(box_num):
		box1 = img_predbox[i, -4:]
		for j in range(i+1, box_num):
			box2 = img_predbox[j, -4:]
			img_iou[i][j] = get_iou(box1, box2)
	return img_iou


# 计算iou偏差之和(f2的值)
def get_sum_iou(select_box, wset, img_iou):
	'''
	select_box: 包含被选择的box信息(图片id box_id bilabel predlabel predscore x1 y1 x2 y2)
	wset: 对应select_box的每个box的下标
	'''
	assert len(select_box) == len(wset)
	diff_iou = 0
	box_num = len(select_box)
	for i in range(box_num):
		box1 = select_box[i]
		for j in range(i+1, box_num):
			box2 = select_box[j]
			bx_l = [box1[3], box2[3]]
			bx_l.sort()
			diff_iou += pow((img_iou[wset[i]][wset[j]] - \
							label_iou[bx_l[0]][bx_l[1]]['mean_iou']), 2) \
							/ label_iou[bx_l[0]][bx_l[1]]['var_iou']
	return diff_iou


# 计算可行域每个取值对应的目标函数f1和f2的值(针对单个labelset)
def get_f1_f2(img_predbox, all_wset, pred_area, k, img_iou, pixeltype):
	'''
	all_wset: 单个labelset对应的可能的w组合，包含w取值为1的box下标
	img_predbox[i]: 图片id box_id bilabel predlabel predscore x1 y1 x2 y2
	'''
	list_f1 = []
	list_f2 = []
	time_f1 = 0
	time_f2 = 0
	for wset in all_wset:
		select_box = []
		for i in wset:
			select_box.append(img_predbox[i])
		select_box = np.asarray(select_box)
		# overlap = get_box_overlap(select_box[:, -4:]) / pred_area
		time1 = time.time()
		overlap = get_pixel_overlap(wset, pixeltype) / pred_area
		f1 = pow((overlap - labelset_overlap[k]['mean_overlap']), 2) \
				 / (labelset_overlap[k]['var_overlap'] + mean_d)
		time2 = time.time()
		f2 = get_sum_iou(select_box, wset, img_iou)
		time3 = time.time()
		time_f1 += time2 - time1
		time_f2 += time3 - time2
		list_f1.append(f1)
		list_f2.append(f2)
	print("time_f1", time_f1)
	print("time_f2", time_f2)
	assert len(all_wset) == len(list_f1) == len(list_f2)
	return list_f1, list_f2


# 获取单个labelset对应的w可能值，即获取可行域
def get_wset(labelset, pred_labelset):
	all_wset = []
	num_pred_boxes = len(pred_labelset)
	num_labels = len(labelset)

	# 保证labelset中没有不存在于pred_labelset的label
	if not set(labelset).issubset(pred_labelset):
		return all_wset
	
	# 对labelset和pred_labelset分别建立dict，key=label，value=box下标集合
	pred_labeldict = defaultdict(list)
	labeldict = defaultdict(list)
	for i in range(num_pred_boxes):
		pred_labeldict[pred_labelset[i]].append(i)
	for i in range(num_labels):
		labeldict[labelset[i]].append(i)

	# 选择box
	select_boxid = []
	for l in labeldict:
		if len(pred_labeldict[l]) < len(labeldict[l]):
			return all_wset
		select_boxid.append(list(combinations(pred_labeldict[l], len(labeldict[l]))))

	wset = list(product(*select_boxid))
	for w in wset:
		wlist = []
		for wi in w:
			wlist += wi
		all_wset.append(sorted(wlist))
	all_wset = np.array(all_wset)
	assert num_labels == all_wset.shape[1]
	return all_wset


# 求解pareto最优解
def get_pareto(list_f1, list_f2, all_wset):
	assert len(list_f1) == len(list_f2) == len(all_wset)
	assert len(list_f1) != 0
	pareto_f1 = []
	pareto_f2 = []
	pareto_w = []
	pareto_index = []	# 保存pareto最优解所在的位置
	
	f1_sorted = np.argsort(list_f1)
	pareto_index.append(f1_sorted[0])
	f2_min_index = np.argmin(list_f2)
	
	for i in f1_sorted[1:]:	# 越往后f1越大
		if list_f2[i] < list_f2[pareto_index[-1]]:
			if list_f1[i] == list_f1[pareto_index[-1]]:
				pareto_index.pop(-1)
			pareto_index.append(i)
		if i == f2_min_index:
			break

	for i in pareto_index:
		pareto_f1.append(list_f1[i])
		pareto_f2.append(list_f2[i])
		pareto_w.append(all_wset[i])
	return pareto_f1, pareto_f2, pareto_w


# f2的值在求解全局pareto最优时用平均值
def modify_iou(f2, w):
	num = len(f2)
	for i in range(num):
		box_num = len(w[i])
		f2[i] = f2[i] / (box_num * (box_num - 1) / 2)
	return f2


# 遍历所有labelset
def labelset_traverse(img_predbox, img_iou):
	'''
	img_predbox: array, shape=(pred_box_num, 9)
	img_predbox[i]: 图片id box_id bilabel predlabel predscore x1 y1 x2 y2
	pred_labelset: faster-rcnn预测的所有box的label集合，array.shape = (pred_box_num,)
	pred_area: faster-rcnn预测的所有box面积的并集
	'''
	assert img_id == img_predbox[0][0]
	pred_labelset = img_predbox[:, 3]
	pred_area = get_box_overlap(img_predbox[:, -4:])
	time1 = time.time()
	pixeltype = get_pixeltype(img_predbox)
	time2 = time.time()
	print(time2 - time1)

	img_pareto_f1 = []
	img_pareto_f2 = []
	img_pareto_w = []
	
	for k in labelset_overlap:
		if k == 'None' or len(k) == 1:
			continue
		labelset = np.array(k)	# shape = (box_num,)
		all_wset = get_wset(labelset, pred_labelset)
		if len(all_wset) == 0:
			continue
		print("k", k, len(all_wset))
		time1 = time.time()
		list_f1, list_f2 = get_f1_f2(img_predbox, all_wset, pred_area, k, img_iou, pixeltype)
		time2 = time.time()
		pareto_f1, pareto_f2, pareto_w = get_pareto(list_f1, list_f2, all_wset)
		time3 = time.time()
		print("f1f2: ", time2 - time1)
		print("pareto: ", time3 - time2)
		print('pareto_f1', pareto_f1)
		print('pareto_f2', pareto_f2)
		print('pareto_w', pareto_w)
		print('--------------------------------------------')
		img_pareto_f1 += pareto_f1
		img_pareto_f2 += pareto_f2
		img_pareto_w += pareto_w
	print(len(img_pareto_f1))
	if len(img_pareto_f1) != 0:
		img_pareto_f2 = modify_iou(img_pareto_f2, img_pareto_w)
		img_pareto_f1, img_pareto_f2, img_pareto_w = get_pareto(img_pareto_f1, img_pareto_f2, img_pareto_w)
	return list(zip(img_pareto_f1, img_pareto_f2, img_pareto_w))


if __name__ == '__main__':
	mode = 'test'
	path = '{}/sgdet_{}'.format(mode, mode)

	# 图片id box_id bilabel predlabel predscore x1 y1 x2 y2
	f = open(os.path.join(path, 'test_predbox_noconstraint.txt'), 'r')
	predbox = []
	img_pareto = dict()
	for line in f.readlines():
		data = line.strip().split(' ')
		data = np.asarray(data, dtype = np.float32)
		if data[0] != img_id:
			img_predbox = np.asarray(predbox)
			print('img_id', img_id)
			img_iou = get_img_iou(img_predbox)
			img_pareto[img_id] = labelset_traverse(img_predbox, img_iou)
			print('+++++++++++++++++++++++++++++++++++++++++')
			print(img_pareto[img_id])
			predbox.clear()
			img_id += 1
			break
		predbox.append(data)
	f.close()

	# with open(os.path.join(path, 'test_pareto_noconstraint.pkl'), 'wb') as fpkl:
	# 	pkl.dump(img_pareto, fpkl)