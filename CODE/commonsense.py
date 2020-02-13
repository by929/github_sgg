import numpy as np
import os


def get_img_info(pkl_path):
	'''
	pkl文件包含：'gt_filenames'：图片名
            'gt_classes': box对应的标签, 从1开始计数, 0表示background
            'gt_relations': (h,t,r) eg.(0,1,38) 0表示第0个box, r从1开始计数, 0表示没有关系
            'gt_boxes': box的坐标
	'''
	imgs_info = pkl.load(open(pkl_path, 'rb'))
	return imgs_info


if __name__=="__main__":
	
	commonsense = dict()
	num_classes = 151
	num_rels = 51

	pkl_path = "train/train_gt_entries.pkl"
	imgs_info = get_img_info(pkl_path)

	for i in range(num_classes):
		commonsense[i] = dict()
		for j in range(num_classes):
			commonsense[i][j] = np.zeros(num_rels)

	for img_info in imgs_info:
		box_classes = img_info['gt_classes']
		rels = img_info['gt_relations']
		for h_id, t_id, r in rels:
			h = box_classes[h_id]
			t = box_classes[t_id]
			commonsense[h][t][r] += 1