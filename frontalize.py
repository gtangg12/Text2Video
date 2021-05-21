import numpy as np
from scipy.spatial import cKDTree
import cv2
import os
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm


from preprocess_img import align_img
from utils import load_lm3d

def frontalize(input_img_pths, mask_s, world_coords_s, reference_world_coords_s, num_imgs, start_frame):
	for n in range(num_imgs):
		# reshape outputs
		img_pth = input_img_pths[n]

		# load images and corresponding 5 facial landmarks
		#img,lm = load_img(file,file.replace('png','txt').replace('jpg','txt'))
		img = Image.open(img_pth)
		lm = all_landmarks[n + start_frame, :5]
		# preprocess input image
		lm3D = load_lm3d()

		input_img,lm_new,transform_params = align_img(img,lm,lm3D)

		#face_shape_ = face_shape_s[n]
		#face_texture_ = face_texture_s[n]
		#face_color_ = face_color_s[n]
		#landmarks_2d_ = landmarks_2d_s[n]
		mask_ = mask_s[n]
		#if not is_windows:
		#	recon_img_ = recon_img_s[n]

		world_coords_ = world_coords_s[n]
		reference_world_coords_ = reference_world_coords_s[n]

		# NN algorithm for pixel querying
		mask_ = mask_.reshape(-1).astype(bool)
		input_img_flat = input_img.reshape(-1, 3)
		world_coords_ = world_coords_.reshape(-1, 3)
		reference_world_coords_ = reference_world_coords_.reshape(-1, 3)

		pixel_values = {}
		for i, x in enumerate(world_coords_):
			if mask_[i]:
				pixel_values[tuple(x)] = input_img_flat[i]
			# get rid of duplicates missed by mask
			# change np.array([-0.008168162, 0.006778449, -0.31529066]) to like world_coords_[0, 0] for other images
			if mask_[i] and np.sum(np.abs(x - world_coords_[0])) < 0.0001:
				mask_[i] = False
		
		masked_coords_ = world_coords_[np.where(mask_)]
		
		"""
		coords_scatterplot_3d(world_coords_, 'world_coords.png', color = input_img_flat[:,::-1]/255)
		coords_scatterplot_3d(reference_world_coords_, 'ref_world_coords.png')
		coords_scatterplot_flatter(world_coords_, 'world_coords_flat.png', color = input_img_flat[:,::-1]/255)
		"""

		'''
		from collections import Counter
		a = [tuple(x) for x in world_coords_[np.where(mask)]]
		print([(tuple(item), count) for item, count in Counter(a).items() if count > 1])
		exit(0)
		'''
		tree = cKDTree(masked_coords_)

		output_img = np.zeros(input_img_flat.shape)
		for i, x in enumerate(reference_world_coords_):
			# if reference image rasterization doesn't hit a mesh for pixel i
			if np.sum(np.abs(x)) == 0:
				continue

			distance, nn_i = tree.query(x)

			#print(nn_i)
			nn = masked_coords_[nn_i]
			#print(distance, x, nn, nn_i, mask[nn_i])
			output_img[i] += pixel_values[tuple(nn)]


		output_img = output_img.reshape(224, 224, 3)
		#print(output_img)

		cv2.imwrite(f'/media/william/DATA/6869/output/frontalized/nHREBzHqFTQ/output_frame_{n+start_frame}.jpg', output_img)

def frontalize_part(process_id):
	start_value = glob_start + jump_size * process_id * 50
	for batch_idx in tqdm(range(jump_size), position=process_id):
		batch_start = batch_idx * 50 + start_value
		mask = np.load(os.path.join(pth, f'masks_{batch_start}_50.npy'))
		#coeffs = np.load(os.path.join(pth, f'coeffs_{batch_start}_150.npy'))
		wcs = np.load(os.path.join(pth, f'wcs_{batch_start}_50.npy'))
		rwcs = np.load(os.path.join(pth, f'rwcs_{batch_start}_50.npy'))
		
		img_pths = [f'/media/william/DATA/6869/nHREBzHqFTQ/frame_{i}.jpg' for i in range(batch_start, batch_start + 150)]
		frontalize(img_pths, mask, wcs, rwcs, 50, batch_start)

if __name__=='__main__':
	pth = '/media/william/DATA/6869/output/nHREBzHqFTQ/'

	num_processes = 1#6
	glob_start = 303#162
	glob_end = 6116
	tot_batches = (glob_end - glob_start) // 50
	jump_size = tot_batches // num_processes 
	all_landmarks = np.load('nHREBzHqFTQ_landmarks.npy').reshape(-1, 25, 2)[:, :5, :]
	with Pool(num_processes) as p:
		p.map(frontalize_part, range(num_processes))
	

