from numpy.ma.core import concatenate
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
import platform
import argparse
from scipy.io import loadmat, savemat
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from preprocess_img import align_img
from utils import *
from face_decoder import Face3D
from options import Option

is_windows = platform.system() == "Windows"

def parse_args():

    desc = "Deep3DFaceReconstruction"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--pretrain_weights', type=str, default=None, help='path for pre-trained model')
    parser.add_argument('--use_pb', type=int, default=1, help='validation data folder')

    return parser.parse_args()

def restore_weights(sess,opt):
	var_list = tf.trainable_variables()
	g_list = tf.global_variables()

	# add batch normalization params into trainable variables
	bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
	bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
	var_list +=bn_moving_vars

	# create saver to save and restore weights
	saver = tf.train.Saver(var_list = var_list)
	saver.restore(sess,opt.pretrain_weights)

def coords_scatterplot_3d(coords, pth, color = None):
	#scatter plots world coordinates of the mask
	ax = Axes3D(plt.figure())
	if color is not None:
		ax.scatter(*(coords.reshape(-1,3).T), facecolors = color)
	else:
		ax.scatter(*(coords.reshape(-1,3).T))
	ax.set_xlabel('x')	
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	plt.savefig(pth)

def coords_scatterplot_flatter(coords, pth, color = None):
	fig, ax = plt.subplots(figsize=(20,20))
	ax.scatter(*((coords.reshape(-1,3).T)[:2]), c = color)
	plt.savefig(pth)

def demo(start_frame, end_frame, plot_frontalized = False):
	# input and output folder
	args = parse_args()

	image_path = 'input'
	save_path = 'output'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	#img_list = glob.glob(image_path + '/' + '*.png')
	#img_list +=glob.glob(image_path + '/' + '*.jpg')
	#img_name = 'william'
	#img_list = [f'xAAmF3H0-ek/frame_{i}.jpg' for i in range(start_frame, end_frame)] #17136 ['input/test1.jpg', 'input/test2.jpg']#
	img_list = [f'/media/william/DATA/6869/nHREBzHqFTQ/frame_{i}.jpg' for i in range(start_frame, end_frame)]
	# read BFM face model
	# transfer original BFM model to our model
	if not os.path.isfile('./BFM/BFM_model_front.mat'):
		transferBFM09()

	# read standard landmarks for preprocessing images
	lm3D = load_lm3d()
	n = 0

	all_landmarks = np.load('nHREBzHqFTQ_landmarks.npy').reshape(-1, 25, 2)

	# build reconstruction model
	with tf.Graph().as_default() as graph:

		with tf.device('/cpu:0'):
			opt = Option(is_train=False)
		opt.batch_size = len(img_list)
		opt.pretrain_weights = args.pretrain_weights
		FaceReconstructor = Face3D()
		images = tf.placeholder(name = 'input_imgs', shape = [opt.batch_size,224,224,3], dtype = tf.float32)

		if args.use_pb and os.path.isfile('network/FaceReconModel.pb'):
			print('Using pre-trained .pb file.')
			graph_def = load_graph('network/FaceReconModel.pb')
			tf.import_graph_def(graph_def,name='resnet',input_map={'input_imgs:0': images})
			# output coefficients of R-Net (dim = 257)
			coeff = graph.get_tensor_by_name('resnet/coeff:0')
		else:
			print('Using pre-trained .ckpt file: %s'%opt.pretrain_weights)
			import networks
			coeff = networks.R_Net(images,is_training=False)

		# reconstructing faces
		FaceReconstructor.Reconstruction_Block(coeff,opt)
		face_shape = FaceReconstructor.face_shape_t
		face_texture = FaceReconstructor.face_texture
		face_color = FaceReconstructor.face_color
		landmarks_2d = FaceReconstructor.landmark_p
		recon_img = FaceReconstructor.render_imgs
		tri = FaceReconstructor.facemodel.face_buf
		mask = FaceReconstructor.img_mask

        # world coords of non frontalized face
		world_coords = FaceReconstructor.world_coords

        # reference coords
		FaceReconstructorReference = Face3D(reference=True)
		FaceReconstructorReference.Reconstruction_Block(coeff,opt)
		reference_world_coords = FaceReconstructorReference.world_coords

		with tf.Session() as sess: #config=tf.ConfigProto(log_device_placement=True)) as sess:
			if not args.use_pb :
				restore_weights(sess,opt)

			print('preprocessing...')
			input_imgs = []
			for idx, file in enumerate(img_list):
				# load images and corresponding 5 facial landmarks
				#img,lm = load_img(file,file.replace('png','txt').replace('jpg','txt'))
				img = Image.open(file)
				lm = all_landmarks[idx + start_frame, :5]
				# preprocess input image
				input_img,lm_new,transform_params = align_img(img,lm,lm3D)
				input_imgs.append(input_img)
			input_imgs = np.concatenate(input_imgs, 0)

			print('calculating...')

			coeff_s,face_shape_s,face_texture_s,face_color_s,landmarks_2d_s,recon_img_s,tri_s, mask_s, world_coords_s, reference_world_coords_s = sess.run([coeff,\
				face_shape,face_texture,face_color,landmarks_2d,recon_img,tri, mask, world_coords, reference_world_coords],feed_dict = {images: input_imgs})

			"""
			np.save('debug_outputs/world_coords', world_coords_s)
			np.save('debug_outputs/reference_world_coords', reference_world_coords_s)
			np.save('debug_outputs/coeffs', coeff_s)

			plt.subplot()
			plt.imshow(mask_s[0,:,:,0])
			plt.savefig('mask')
			"""

			print(world_coords_s.shape)

			#np.savetxt("world_coords_.txt", world_coords_, newline=" ", fmt='%1.2f')
			#np.set_printoptions(threshold=np.nan, precision=5)
			print(world_coords_s[:, 70:75, 70:75])
			print(reference_world_coords_s[:, 70:75, 70:75])
			#exit(0)
			if not plot_frontalized:
				return coeff_s, mask_s, world_coords_s, reference_world_coords_s

			print('reconstructing...')
			for n in range(len(img_list)):
				# reshape outputs
				input_img = input_imgs[n]
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

				plt.subplot()
				plt.imshow(output_img[:,:,::-1].astype(int))
				plt.savefig(f'output_frame_{n+start_frame}.jpg')
				
				
			#exit(0)

			"""
			# save output files
			if not is_windows:
				savemat(os.path.join(save_path,file.split(os.path.sep)[-1].replace('.png','.mat').replace('jpg','mat')),{'cropped_img':input_img[:,:,::-1],'recon_img':recon_img_,'coeff':coeff_,\
					'face_shape':face_shape_,'face_texture':face_texture_,'face_color':face_color_,'lm_68p':landmarks_2d_,'lm_5p':lm_new})
			save_obj(os.path.join(save_path,file.split(os.path.sep)[-1].replace('.png','_mesh.obj').replace('.jpg','_mesh.obj')),face_shape_,tri_,np.clip(face_color_,0,255)/255) # 3D reconstruction face (in canonical view)
			"""
				
if __name__ == '__main__':
	start_frame = 303
	end_frame = 6117
	batch_size = 50
	
	for s in range(start_frame, end_frame-batch_size, batch_size):
		c, m, wc, rwc = demo(s, s+batch_size, plot_frontalized=False)
		np.save(f'/media/william/DATA/6869/output/nHREBzHqFTQ/coeffs_{s}_{batch_size}', c)		
		np.save(f'/media/william/DATA/6869/output/nHREBzHqFTQ/masks_{s}_{batch_size}', m)
		np.save(f'/media/william/DATA/6869/output/nHREBzHqFTQ/wcs_{s}_{batch_size}', wc)
		np.save(f'/media/william/DATA/6869/output/nHREBzHqFTQ/rwcs_{s}_{batch_size}', rwc)
	
	#demo(15, 90, True)

