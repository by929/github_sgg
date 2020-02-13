import dill as pkl
import numpy as np
import os

box_id = -1
img_id = 0
box_faster_cnt = 0
box_motif_cnt = 0

# # 校验测试集是否正确
# def generate_imgnames():
# 	gt_img_info = pkl.load(open("../motif_gt_entries.pkl", 'rb'))
# 	img_names = []
# 	for img_info in gt_img_info:
# 		img_names.append(img_info['gt_filenames'])
# 	return img_names
# img_names = generate_imgnames()

def generate_filenames(mode, start_i):
	filenames = []
	if mode == 'test':
		for i in range(start_i, 26000, 1000):
			filename = 'vg_{}_{}-{}.pkl'.format(mode, i, i+1000)
			filenames.append(filename)
		filenames.append('vg_test_26000-26446.pkl')
	elif mode == 'train':
		for i in range(start_i, 57000, 1000):
			filename = 'vg_{}_{}-{}.pkl'.format(mode, i, i+1000)
			filenames.append(filename)
		filenames.append('vg_train_57000-57723.pkl')
		# np.random.shuffle(filenames)
	return filenames

def write2txt(img_info, f):
	global box_id, img_id, box_faster_cnt, box_motif_cnt
	# assert img_names[img_id] == img_info['img_name']
	feamap = img_info['pred_boxes_fmap'].cpu().numpy()
	box_num = img_info['gt_classes'].shape[0]

	box_faster_cnt += sum(img_info['gt_classes'] == img_info['pred_obj_label'])
	box_motif_cnt += sum(img_info['gt_classes'] == img_info['motif_pred_classes'])

	h, w, scale = img_info['im_sizes']
	h = h / scale
	w = w / scale
	scale_new = 1024 / max(h, w)
	h = h * scale_new
	w = w * scale_new

	for i in range(box_num):
		box_id += 1
		f.write(str(img_id) + ' ')
		f.write(str(box_id) + ' ')
		f.write(str(img_info['gt_classes'][i]) + ' ')
		f.write(str(img_info['pred_obj_label'][i]) + ' ')
		f.write(str(img_info['pred_obj_score'][i]) + ' ')
		f.write(str(img_info['motif_pred_classes'][i]) + ' ')
		f.write(str(img_info['motif_pred_obj_scores'][i]) + ' ')
		for num in feamap[i]:
			f.write(str(round(num, 4)) + ' ')
		for num in img_info['pred_obj_score_all'][i]:
			f.write(str(round(num, 4)) + ' ')
		for num in img_info['gt_boxes'][i]:
			f.write(str(round(num, 4)) + ' ')
		f.write(str(round(h, 4)) + ' ' + str(round(w, 4)) + '\n')
	img_id += 1


if __name__=="__main__":
	mode = 'train'
	filenames = generate_filenames(mode, 0)
	# filenames = ['vg_test_0-1000.pkl']

	path = '{}/{}_sgcls_pkl'.format(mode, mode)
	save_path = '{}/{}_sgcls_txt'.format(mode, mode)

	for filename in filenames:
		print(filename)
		fn, ex = os.path.splitext(filename)
		save_name = fn + '.txt'
		save_file = open(os.path.join(save_path, save_name), 'w')
		img_infos = pkl.load(open(os.path.join(path, filename), 'rb'))
		
		for img_info in img_infos:
			write2txt(img_info, save_file)
		save_file.close()
		print(filename, box_id, box_faster_cnt, box_motif_cnt)