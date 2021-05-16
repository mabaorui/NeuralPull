# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:44:22 2020

@author: Administrator
"""

import numpy as np
import tensorflow as tf 
import os 
import shutil
import random
import math
import scipy.io as sio
import time
from skimage import measure
import binvox_rw
import argparse
import trimesh
from im2mesh.utils import libmcubes
from im2mesh.utils.libkdtree import KDTree
import re

parser = argparse.ArgumentParser()
parser.add_argument('--train',action='store_true', default=False)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--class_idx', type=str, default="026911156")
parser.add_argument('--save_idx', type=int, default=-1)
parser.add_argument('--CUDA', type=int, default=0)
parser.add_argument('--dataset', type=str, default="shapenet")
a = parser.parse_args()

cuda_idx = str(a.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx


BS = 1
POINT_NUM = 5000
POINT_NUM_GT = 20000
INPUT_DIR = a.data_dir
#INPUT_DIR = '/home/mabaorui/AtlasNetOwn/data/sphere/'
OUTPUT_DIR = a.out_dir
if(a.dataset=="shapenet"):
    GT_DIR = './data/ShapeNet_GT/' + a.class_idx + '/'
if(a.dataset=="famous"):
    GT_DIR = './data/famous_noisefree/03_meshes/'
if(a.dataset=="ABC"):
    GT_DIR = './data/abc_noisefree/03_meshes/'

TRAIN = a.train
bd = 0.55

if(TRAIN):
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print ('test_res_dir: deleted and then created!')
    os.makedirs(OUTPUT_DIR)


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

#        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
#        # Handle normals that point into wrong direction gracefully
#        # (mostly due to mehtod not caring about this in generation)
#        normals_dot_product = np.abs(normals_dot_product)
        
        normals_dot_product = np.abs(normals_tgt[idx] * normals_src)
        normals_dot_product = normals_dot_product.sum(axis=-1)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product

def eval_pointcloud(pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''
        # Return maximum losses if pointcloud is empty


        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()
        #print(completeness,accuracy,completeness2,accuracy2)
        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        print('chamferL2:',chamferL2)
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)
        print('normals_correctness:',normals_correctness,'chamferL1:',chamferL1)
        return normals_correctness, chamferL1, chamferL2

def safe_norm_np(x, epsilon=1e-12, axis=1):
    return np.sqrt(np.sum(x*x, axis=axis) + epsilon)

def safe_norm(x, epsilon=1e-12, axis=None):
  return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)

def boundingbox(x,y,z):
    return min(x),max(x),min(y),max(y),min(z),max(z)

    

def chamfer_distance_tf_None(array1, array2):
    array1 = tf.reshape(array1,[-1,3])
    array2 = tf.reshape(array2,[-1,3])
    av_dist1 = av_dist_None(array1, array2)
    av_dist2 = av_dist_None(array2, array1)
    return av_dist1+av_dist2

def distance_matrix_None(array1, array2, num_point, num_features = 3):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
            , it's size: (num_point, num_point)
    """
    expanded_array1 = tf.tile(array1, (num_point, 1))
    expanded_array2 = tf.reshape(
            tf.tile(tf.expand_dims(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = tf.norm(expanded_array1-expanded_array2, axis=1)
    distances = tf.reshape(distances, (num_point, num_point))
    return distances

def av_dist_None(array1, array2):
    """
    arguments:
        array1, array2: both size: (num_points, num_feature)
    returns:
        distances: size: (1,)
    """
    distances = distance_matrix_None(array1, array2,points_input_num[0,0])
    distances = tf.reduce_min(distances, axis=1)
    distances = tf.reduce_mean(distances)
    return distances



files = []
files_path = []

if(a.dataset == "shapenet"):
    f = open('./data/shapenet_val.txt','r')
    for index,line in enumerate(f):
        if(line.strip().split('/')[0]==a.class_idx):
            #print(line)
            files.append(line.strip().split('/')[1])
    f.close()

if(a.dataset == "famous"):
    f = open('./data/famous_testset.txt','r')
    for index,line in enumerate(f):
        #print(line)
        files.append(line.strip('\n'))
    f.close()
    
if(a.dataset == "ABC" or a.dataset == "other"):
    fileAll = os.listdir(INPUT_DIR)
    for file in fileAll:
        if(re.findall(r'.*.npz', file, flags=0)):
            print(file.strip().split('.')[0])
            files.append(file.strip().split('.')[0])

for file in files:
    files_path.append(INPUT_DIR + file + '.npz')
SHAPE_NUM = len(files_path)
print('SHAPE_NUM:',SHAPE_NUM)

pointclouds = []
samples = []
mm = 0
if(TRAIN):
    for file in files_path:
        # if(mm>10):
        #     break
        # mm = mm + 1
        #print(INPUT_DIR + file + '.npz')
        load_data = np.load(file)
        #print(load_data['sample_near'].shape)
        point = np.asarray(load_data['sample_near']).reshape(-1,POINT_NUM,3)
        sample = np.asarray(load_data['sample']).reshape(-1,POINT_NUM,3)
        pointclouds.append(point)
        samples.append(sample)
    
    pointclouds = np.asarray(pointclouds)
    samples = np.asarray(samples)
    print('data shape:',pointclouds.shape,samples.shape)

feature = tf.placeholder(tf.float32, shape=[BS,None,SHAPE_NUM])
points_target = tf.placeholder(tf.float32, shape=[BS,None,3])
input_points_3d = tf.placeholder(tf.float32, shape=[BS, None,3])
points_target_num = tf.placeholder(tf.int32, shape=[1,1])
points_input_num = tf.placeholder(tf.int32, shape=[1,1])



feature_f = tf.nn.relu(tf.layers.dense(feature,128))
net = tf.nn.relu(tf.layers.dense(input_points_3d, 512))
net = tf.concat([net,feature_f],2)
print('net:',net)
with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
    for i in range(8):
        with tf.variable_scope("resnetBlockFC_%d" % i ):
            b_initializer=tf.constant_initializer(0.0)
            w_initializer = tf.random_normal_initializer(mean=0.0,stddev=np.sqrt(2) / np.sqrt(512))
            net = tf.layers.dense(tf.nn.relu(net),512,kernel_initializer=w_initializer,bias_initializer=b_initializer)
            
b_initializer=tf.constant_initializer(-0.5)
w_initializer = tf.random_normal_initializer(mean=2*np.sqrt(np.pi) / np.sqrt(512), stddev = 0.000001)
print('net:',net)
sdf = tf.layers.dense(tf.nn.relu(net),1,kernel_initializer=w_initializer,bias_initializer=b_initializer)
print('sdf',sdf)

grad = tf.gradients(ys=sdf, xs=input_points_3d) 
print('grad',grad)
print(grad[0])
normal_p_lenght = tf.expand_dims(safe_norm(grad[0],axis = -1),-1)
print('normal_p_lenght',normal_p_lenght)
grad_norm = grad[0]/normal_p_lenght
print('grad_norm',grad_norm)

g_points = input_points_3d - sdf * grad_norm

#loss = tf.losses.huber_loss(points_target, g_points)
#loss = chamfer_distance_tf_None(point_target_near, g_points)
#loss = chamfer_distance_tf_None(points_target, g_points)
l2_loss = tf.norm((points_target-g_points), axis=-1)
print('l2_loss:',l2_loss)
loss = tf.reduce_mean(l2_loss)


t_vars = tf.trainable_variables()
optim = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9)
loss_grads_and_vars = optim.compute_gradients(loss, var_list=t_vars)
loss_optim = optim.apply_gradients(loss_grads_and_vars)


config = tf.ConfigProto(allow_soft_placement=False) 
saver_restore = tf.train.Saver(var_list=t_vars)
saver = tf.train.Saver(max_to_keep=2000000)



with tf.Session(config=config) as sess:
    feature_bs = []
    for i in range(SHAPE_NUM):
        tt = []
        for j in range(int(POINT_NUM)):
            t = np.zeros(SHAPE_NUM)
            t[i] = 1
            tt.append(t)
        feature_bs.append(tt)
    feature_bs = np.asarray(feature_bs)
    if(TRAIN):
        print('train start')
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        
        POINT_NUM_GT_bs = np.array(POINT_NUM_GT).reshape(1,1)
        points_input_num_bs = np.array(POINT_NUM).reshape(1,1)
        
        
        for i in range(40000):
            #start_time = time.time()
            epoch_index = np.random.choice(SHAPE_NUM, SHAPE_NUM, replace = False)
            #epoch_index = np.random.choice(10, 10, replace = False)
            loss_i = 0
            for epoch in epoch_index:
                rt = random.randint(0,samples.shape[1]-1)
                input_points_2d_bs = samples[epoch,rt,:,:].reshape(BS, POINT_NUM, 3)
                point_gt = pointclouds[epoch,rt,:,:].reshape(BS,POINT_NUM,3)
                feature_bs_t = feature_bs[epoch,:,:].reshape(1,-1,SHAPE_NUM)
                _,loss_c = sess.run([loss_optim,loss],feed_dict={input_points_3d:input_points_2d_bs,points_target:point_gt,feature:feature_bs_t,points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs})
                loss_i = loss_i + loss_c
            loss_i = loss_i / SHAPE_NUM
            if(i%10 == 0):
                print('epoch:', i, 'epoch loss:', loss_i)
            if(i%500 == 0):
                print('save model')
                saver.save(sess, os.path.join(OUTPUT_DIR, "model"), global_step=i+1)
        end_time = time.time()
        print('run_time:',end_time-start_time)
    else:
        print('test start')
        checkpoint = tf.train.get_checkpoint_state(OUTPUT_DIR).all_model_checkpoint_paths
        print(checkpoint[a.save_idx])
        saver.restore(sess, checkpoint[a.save_idx])
        s = np.arange(-bd,bd, (2*bd)/128)
            
        print(s.shape[0])
        vox_size = s.shape[0]
        POINT_NUM_GT_bs = np.array(vox_size).reshape(1,1)
        points_input_num_bs = np.array(POINT_NUM).reshape(1,1)
        input_points_2d_bs = []
        for i in s:
            for j in s:
                for k in s:
                    input_points_2d_bs.append(np.asarray([i,j,k]))
        input_points_2d_bs = np.asarray(input_points_2d_bs)
        print('input_points_2d_bs',input_points_2d_bs.shape)
        input_points_2d_bs = input_points_2d_bs.reshape((vox_size,vox_size,vox_size,3))
        POINT_NUM_GT_bs = np.array(vox_size*vox_size).reshape(1,1)

        test_num = SHAPE_NUM
        #test_num = 4
        print('test_num:',test_num)
        cd = 0
        nc = 0
        cd2 = 0
        #for epoch in range(20):
        for epoch in range(test_num):
            print('test:',epoch)
#            if(os.path.exists(OUTPUT_DIR + file_test[epoch] + '.npz')):
#                print('exit')
#                continue
#            with open(OUTPUT_DIR + 'gt_' + files[START+epoch] + '.txt'.format(epoch),'w') as f:
#                for i in range(pointclouds.shape[1]):
#                    x = pointclouds[epoch,i,0]
#                    y = pointclouds[epoch,i,1]
#                    z = pointclouds[epoch,i,2]
#                    f.write(str(x)+';')
#                    f.write(str(y)+';')
#                    f.write(str(z)+';\n')  
           
            
                        
            vox = []
            feature_bs = []
            for j in range(vox_size*vox_size):
                t = np.zeros(SHAPE_NUM)
                t[epoch] = 1
                feature_bs.append(t)
            feature_bs = np.asarray(feature_bs)
            for i in range(vox_size):
                
                input_points_2d_bs_t = input_points_2d_bs[i,:,:,:]
                input_points_2d_bs_t = input_points_2d_bs_t.reshape(BS, vox_size*vox_size, 3)
                feature_bs_t = feature_bs.reshape(BS,vox_size*vox_size,SHAPE_NUM)
                sdf_c = sess.run([sdf],feed_dict={input_points_3d:input_points_2d_bs_t,feature:feature_bs_t,points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs})
                vox.append(sdf_c)
                
            vox = np.asarray(vox)
            #print('vox',vox.shape)
            vox = vox.reshape((vox_size,vox_size,vox_size))
            vox_max = np.max(vox.reshape((-1)))
            vox_min = np.min(vox.reshape((-1)))
            print('max_min:',vox_max,vox_min)
            
            threshs = [0.005]
            for thresh in threshs:
                print(np.sum(vox>thresh),np.sum(vox<thresh))
                
                if(np.sum(vox>0.0)<np.sum(vox<0.0)):
                    thresh = -thresh
                print('model:',epoch,'thresh:',thresh)
                vertices, triangles = libmcubes.marching_cubes(vox, thresh)
                if(vertices.shape[0]<10 or triangles.shape<10):
                    print('no sur---------------------------------------------')
                    continue
                if(np.sum(vox>0.0)>np.sum(vox<0.0)):
                    triangles_t = []
                    for it in range(triangles.shape[0]):
                        tt = np.array([triangles[it,2],triangles[it,1],triangles[it,0]])
                        triangles_t.append(tt)
                    triangles_t = np.asarray(triangles_t)
                else:
                    triangles_t = triangles
                    triangles_t = np.asarray(triangles_t)

                vertices -= 0.5
                # Undo padding
                vertices -= 1
                # Normalize to bounding box
                vertices /= np.array([vox_size-1, vox_size-1, vox_size-1])
                vertices = 1.1 * (vertices - 0.5)
                mesh = trimesh.Trimesh(vertices, triangles_t,
                               vertex_normals=None,
                               process=False)
                mesh.export(OUTPUT_DIR +  '/occn_' + files[epoch] + '_'+ str(thresh) + '.off')
                
    
                mesh = trimesh.Trimesh(vertices, triangles,
                                   vertex_normals=None,
                                   process=False)
                if(a.dataset == 'other'):
                    continue
                if(a.dataset=="shapenet"):
                    ps, idx = mesh.sample(1000000, return_index=True)
                else:
                    ps, idx = mesh.sample(10000, return_index=True)
                ps = ps.astype(np.float32)
                normals_pred = mesh.face_normals[idx]
                
                if(a.dataset=="shapenet"):
                    data = np.load(GT_DIR + files[epoch] + '/pointcloud.npz')
                    #data = np.load(file_test[epoch])
                    pointcloud = data['points']
                    normal = data['normals']
                else:
                    mesh_gt = trimesh.load(GT_DIR + files[epoch] + '.ply')
                    pointcloud, idx_gt = mesh_gt.sample(10000, return_index=True)
                    pointcloud = pointcloud.astype(np.float32)
                    normal = mesh_gt.face_normals[idx_gt]
                
                nc_t,cd_t,cd2_t = eval_pointcloud(ps,pointcloud.astype(np.float32),normals_pred.astype(np.float32),normal.astype(np.float32))
                np.savez(OUTPUT_DIR + files[epoch]+ '_'+ str(thresh),pp = ps, np = normals_pred, p = pointcloud, n = normal, nc = nc_t, cd = cd_t, cd2 = cd2_t)
                nc = nc + nc_t
                cd = cd + cd_t
                cd2 = cd2 + cd2_t
        #print('mean_nc:',nc/20,'mean_cd:',cd/20,'cd2:',cd2/20)
        print('mean_nc:',nc/test_num,'mean_cd:',cd/test_num,'cd2:',cd2/test_num)
                    
    
    