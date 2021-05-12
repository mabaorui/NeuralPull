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


parser = argparse.ArgumentParser()
parser.add_argument('--train',action='store_true', default=False)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--class_idx', type=str, default="026911156")
parser.add_argument('--save_idx', type=int, default=-1)
parser.add_argument('--CUDA', type=int, default=0)
parser.add_argument('--class_name', type=int, default=0)
a = parser.parse_args()

cuda_idx = str(a.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx

BS = 1
POINT_NUM = 5000
POINT_NUM_GT = 20000
train_list = '/data/mabaorui/AtlasNetOwn/data/shapenetcore_ids/' + a.class_idx + '_trainids.txt'
val_list = '/data/mabaorui/AtlasNetOwn/data/shapenetcore_ids/' + a.class_idx + '_valids.txt'
test_list = '/data/mabaorui/AtlasNetOwn/data/shapenetcore_ids/' + a.class_idx + '_testids.txt'
INPUT_DIR = a.data_dir
OUTPUT_DIR = a.out_dir
t = np.load('/data/mabaorui/AtlasNetOwn/data/features/' + a.class_name + '.npz')
features_train = np.concatenate((t['train'],t['val']),axis = 0)
TRAIN = a.train
bd = 0.55


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
        print(completeness,accuracy,completeness2,accuracy2)
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

def get_data_from_filename(filename):
    load_data = np.load(filename)
    point = np.asarray(load_data['sample_near']).reshape(-1,POINT_NUM,3)
    sample = np.asarray(load_data['sample']).reshape(-1,POINT_NUM,3)
    rt = random.randint(0,sample.shape[0]-1)
    #rt = random.randint(0,int((sample.shape[0]-1)/5))
    sample = sample[rt,:,:].reshape(BS, POINT_NUM, 3)
    point = point[rt,:,:].reshape(BS, POINT_NUM, 3)

    
    
    #print('input_points_bs:',filename)
    #print(input_points_bs)
    return point.astype(np.float32), sample.astype(np.float32)

files = []
files_path = []
if(TRAIN):
    f = open(train_list,'r')
    for index,line in enumerate(f):
        files.append(line.strip().split('/')[1])
    f.close()
    
    f = open(val_list,'r')
    for index,line in enumerate(f):
        files.append(line.strip().split('/')[1])
    f.close()
else:
    f = open(test_list,'r')
    for index,line in enumerate(f):
        files.append(line.strip().split('/')[1])
    f.close()
    
for file in files:
    #print(INPUT_DIR + file)
    files_path.append(INPUT_DIR + file + '.npz')
SHAPE_NUM = len(files_path)
print('SHAPE_NUM:',SHAPE_NUM)
  
filelist = tf.placeholder(tf.string, shape=[None])

ds = tf.data.Dataset.from_tensor_slices((filelist))

ds = ds.map(
    lambda item: tuple(tf.py_func(get_data_from_filename, [item], (tf.float32, tf.float32))),num_parallel_calls = 4)
ds = ds.repeat()  # Repeat the input indefinitely.
ds = ds.batch(1)
ds = ds.prefetch(buffer_size = 50)
iterator = ds.make_initializable_iterator()
next_element = iterator.get_next()

feature = tf.placeholder(tf.float32, shape=[BS,None,512])
if(TRAIN):
    points_target = tf.reshape(next_element[0],[BS,-1,3])
    input_points_3d = tf.reshape(next_element[1],[BS,-1,3])
else:
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

#loss = tf.losses.huber_loss(point_target_near, g_points)
#loss = chamfer_distance_tf_None(point_target_near, g_points)
loss = chamfer_distance_tf_None(points_target, g_points)


t_vars = tf.trainable_variables()
optim = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9)
loss_grads_and_vars = optim.compute_gradients(loss, var_list=t_vars)
loss_optim = optim.apply_gradients(loss_grads_and_vars)


config = tf.ConfigProto(allow_soft_placement=False) 
saver_restore = tf.train.Saver(var_list=t_vars)
saver = tf.train.Saver(max_to_keep=2000000)


with tf.Session(config=config) as sess:
     
    if(TRAIN):
        print('train start')
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        
        POINT_NUM_GT_bs = np.array(POINT_NUM_GT).reshape(1,1)
        points_input_num_bs = np.array(POINT_NUM).reshape(1,1)
        for i in range(4000000):
            epoch_index = np.random.choice(SHAPE_NUM, SHAPE_NUM, replace = False)
            ini_data_path_epoch = []
            for fi in range(SHAPE_NUM):
                ini_data_path_epoch.append(files_path[epoch_index[fi]])
            sess.run(iterator.initializer, feed_dict={filelist: ini_data_path_epoch})
            loss_i = 0
            for epoch in epoch_index:
                img_index = random.randint(0,23)
                net_c = features_train[epoch*24+img_index,:]
                net_c = net_c.reshape((-1))
                net_c = np.expand_dims(net_c,0).repeat(POINT_NUM,axis=0)
                net_c = net_c.reshape((1,POINT_NUM, 512))
                feature_bs_t = net_c 
                _, loss_c = sess.run([loss_optim,loss],feed_dict={feature:feature_bs_t,points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs})
                loss_i = loss_i + loss_c
            loss_i = loss_i / SHAPE_NUM
            with open(OUTPUT_DIR + '/loss.txt','a') as f:
                f.write(str(loss_i)+';')
            if(i%100 == 0):
                print('save model')
                #print(p_points_bs)
                saver.save(sess, os.path.join(OUTPUT_DIR, "model"), global_step=i+1)

        
        end_time = time.time()
        print('run_time:',end_time-start_time)
    else:
        pointclouds = []
        samples = []
        normals_gt = []
        
        files = []
        
        f = open(test_list,'r')
        for index,line in enumerate(f):
            files.append(line.strip().split('/')[1])
        f.close()
        
        
        for file in files:
            
            data = np.load(INPUT_DIR + file + '/pointcloud.npz')
        
            pointcloud = data['points'].reshape(1,-1,3)
            normal = data['normals'].reshape(1,-1,3)
            pointcloud = pointcloud.reshape(1,-1,3)
            pointclouds.append(pointcloud)
            normals_gt.append(normal)
            
        t = np.load('/data/mabaorui/AtlasNetOwn/data/features/' + a.class_name + '.npz')
        feature_test = t['test']
        print('test start')
        checkpoint = tf.train.get_checkpoint_state(OUTPUT_DIR).all_model_checkpoint_paths
        print(checkpoint[-1])
        saver.restore(sess, checkpoint[-1])
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
        print('test_num:',test_num)
        cd = 0
        nc = 0
        for epoch in range(0,test_num):
            with open(INPUT_DIR + 'gt_' + files[epoch] + '.txt','w') as f:
                for i in range(pointclouds.shape[1]):
                    x = pointclouds[epoch,i,0]
                    y = pointclouds[epoch,i,1]
                    z = pointclouds[epoch,i,2]
                    f.write(str(x)+';')
                    f.write(str(y)+';')
                    f.write(str(z)+';\n')  
           
            nc_best = 10000000
            cd_best = 10000000
            #threshs = [-0.01,0.0,0.01]
            threshs = [0.01]
            for thresh in threshs:
                #print('thresh:',thresh)
                for img_index in range(24):            
                    vox = []
                    
                    for i in range(vox_size):
                        input_points_2d_bs_t = input_points_2d_bs[i,:,:,:]
                        input_points_2d_bs_t = input_points_2d_bs_t.reshape(BS, vox_size*vox_size, 3)
                        net_c = feature_test[epoch*24+img_index,:]
                        net_c = net_c.reshape((-1))
                        #print(net_c.shape)
                        net_c = np.expand_dims(net_c,0).repeat(vox_size*vox_size,axis=0)
                        #print(net_c.shape)
                        net_c = net_c.reshape((1,vox_size*vox_size, 512))
                        #print(net_c)
                        feature_bs_t = net_c
                        sdf_c = sess.run([sdf],feed_dict={input_points_3d:input_points_2d_bs_t,feature:feature_bs_t,points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs})
                        vox.append(sdf_c)
                        
                    vox = np.asarray(vox)
                    #print('vox',vox.shape)
                    vox = vox.reshape((vox_size,vox_size,vox_size))
                    vox_max = np.max(vox.reshape((-1)))
                    vox_min = np.min(vox.reshape((-1)))
                    #print('thresh:',thresh)
                    print('max_min:',vox_max,vox_min)
                    #print(abs(vox_max),abs(vox_min))
                    if(abs(vox_max)>abs(vox_min)):
                        thresh = 0.01
                    else:
                        thresh = -0.01
                    print('thresh:',thresh)
                    #print(np.sum(vox>thresh),np.sum(vox<thresh))
                    
                    vertices, triangles = libmcubes.marching_cubes(vox, thresh)
                    if(vertices.shape[0]<10 or triangles.shape<10):
                        print('no sur---------------------------------------------')
                        continue
                    if(np.sum(vox>thresh)>np.sum(vox<thresh)):
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
                    #mesh.export(OUTPUT_DIR +  '/occn_' + files[epoch] + '_'+ str(thresh) + '_{}.off'.format(img_index))
                    mesh.export(OUTPUT_DIR +  '/occn_' + files[epoch] + '_'+ '0.01' + '_{}.off'.format(img_index))
                    
                    
                    
                    mesh = trimesh.Trimesh(vertices, triangles,
                                       vertex_normals=None,
                                       process=False)
                    ps, idx = mesh.sample(100000, return_index=True)
                    ps = ps.astype(np.float32)
                    normals_pred = mesh.face_normals[idx]
                    
        
                    nc_t,cd_t,cd2_t = eval_pointcloud(ps,pointclouds[epoch,:,:].astype(np.float32),normals_pred.astype(np.float32),normals_gt[epoch,:,:].astype(np.float32))
                    #np.savez(OUTPUT_DIR + files[epoch] + '_{}'.format(img_index),pp = ps, np = normals_pred, p = pointcloud[0,:,:], n = normal[0,:,:], nc = nc_t, cd = cd_t, cd2 = cd2_t)
                    #np.savez(OUTPUT_DIR + files[epoch] + '_'+ str(thresh) + '_{}'.format(img_index), nc = nc_t, cd = cd_t, cd2 = cd2_t)
                    np.savez(OUTPUT_DIR + files[epoch] + '_'+ '0.01' + '_{}'.format(img_index), nc = nc_t, cd = cd_t, cd2 = cd2_t)
                    if(cd_t<cd_best):
                        cd_best = cd_t
                        nc_best = nc_t
                        print('model:',epoch,'nc_best:',nc_best,'cd_best:',cd_best)
                nc = nc + nc_best
                cd = cd + cd_best
        print('mean_nc:',nc/test_num,'mean_cd:',cd/test_num)
                    
    
    