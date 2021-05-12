# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:44:22 2020

@author: Administrator
"""
#############
#多个模型one hot向量 + 读自己数据 + sdf 导数 cd + 与查询点的最近点算回归 + 多模型 + 测试代码 + feature + occ marching cube + 测试集*** + 多个阈值测试
#################
import numpy as np
import tensorflow as tf 
import os 
import shutil
import random
import math
import scipy.io as sio
import time
#from tf_grouping import query_ball_point, group_point, knn_point
from skimage import measure
import binvox_rw
from im2mesh.utils import libmcubes
import trimesh
import argparse
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
POINT_NUM_GT = 100000

train_list = '/data/mabaorui/AtlasNetOwn/data/shapenetcore_ids/' + a.class_idx + '_trainids.txt'
val_list = '/data/mabaorui/AtlasNetOwn/data/shapenetcore_ids/' + a.class_idx + '_valids.txt'
INPUT_DIR = a.data_dir
OUTPUT_DIR = a.out_dir
t = np.load('/data/mabaorui/AtlasNetOwn/data/features/' + a.class_name + '.npz')
features_train = np.concatenate((t['train'],t['val']),axis = 0)
TRAIN = a.train
bd = 0.55

train_list = '/data/mabaorui/AtlasNetOwn/data/shapenetcore_ids/04256520_trainids.txt'
val_list = '/data/mabaorui/AtlasNetOwn/data/shapenetcore_ids/04256520_trainids.txt'
test_list = '/data/mabaorui/AtlasNetOwn/data/shapenetcore_ids/04256520_testids.txt'
INPUT_DIR = '/data/mabaorui/project4/data/ShapeNet/04256520/'
#INPUT_DIR = '/home/mabaorui/AtlasNetOwn/data/sphere/'
OUTPUT_DIR = '/data1/mabaorui/AtlasNetOwn/04256520_train_5000_0001/'
t = np.load('/data/mabaorui/AtlasNetOwn/data/features/sofa.npz')
feature_test = t['test']
SHAPE_NUM = 40000
BD_EMPTY = 0.05
TRAIN = a.train
bd = 0.55

#if(TRAIN):
#    if os.path.exists(OUTPUT_DIR):
#        shutil.rmtree(OUTPUT_DIR)
#        print ('test_res_dir: deleted and then created!')
#    os.makedirs(OUTPUT_DIR)

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

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
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

        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)
        print('normals_correctness:',normals_correctness,'chamferL1:',chamferL1,'chamferL2:',chamferL2)
        return normals_correctness, chamferL1, chamferL2

def read_sdf(file_path):

    with open(file_path) as file:  
        line = file.readline()

        # Get grid resolutions
        grid_res = line.split()
        grid_res_x = int(grid_res[0])
        grid_res_y = int(grid_res[1])
        grid_res_z = int(grid_res[2])
        print('res:',grid_res_x,grid_res_y,grid_res_z)

        # Get bounding box min
        line = file.readline()
        bounding_box_min = line.split()
        bounding_box_min_x = float(bounding_box_min[0]) 
        bounding_box_min_y = float(bounding_box_min[1])
        bounding_box_min_z = float(bounding_box_min[2]) 
        print(bounding_box_min_x,bounding_box_min_y,bounding_box_min_z)

        line = file.readline()
        voxel_size = float(line)
        print('voxel_size:',voxel_size)

        # max bounding box (we need to plus 0.0001 to avoid round error)
        bounding_box_max_x = bounding_box_min_x + voxel_size * (grid_res_x - 1)
        bounding_box_max_y = bounding_box_min_y + voxel_size * (grid_res_y - 1) 
        bounding_box_max_z = bounding_box_min_z + voxel_size * (grid_res_z - 1) 

        min_bounding_box_min = min(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z) 
        # print(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z)
        max_bounding_box_max = max(bounding_box_max_x, bounding_box_max_y, bounding_box_max_z) 
        # print(bounding_box_max_x, bounding_box_max_y, bounding_box_max_z)
        max_dist = max(bounding_box_max_x - bounding_box_min_x, bounding_box_max_y - bounding_box_min_y, bounding_box_max_z - bounding_box_min_z)

        # max_dist += 0.1
        max_grid_res = max(grid_res_x, grid_res_y, grid_res_z)

        grid = []
        for i in range(grid_res_x):
            grid.append([])
            for j in range(grid_res_y):
                grid[i].append([])
                for k in range(grid_res_z):
#                    for l in range(3):
#                        grid[i][j].append(0)
                    # grid_value = float(file.readline())
                    grid[i][j].append([0,0,0])
                    # lst.append(grid_value)
          
        grid_value = []
        for i in range(grid_res_x):
            grid_value.append([])
            for j in range(grid_res_y):
                grid_value[i].append([])
                for k in range(grid_res_z):
                    # grid_value = float(file.readline())
                    grid_value[i][j].append(2)
                    # lst.append(grid_value)
        for i in range(grid_res_z):
            for j in range(grid_res_y):
                for k in range(grid_res_x):
                    tt = float(file.readline())
                    grid_value[k][j][i] = tt             
        for i in range(grid_res_x):
            for j in range(grid_res_y):
                for k in range(grid_res_z):
                    tt = [bounding_box_min_x + voxel_size * i,bounding_box_min_y + voxel_size * j,bounding_box_min_z + voxel_size * k]
                    grid[i][j][k] = tt
        print('done')
        grid = np.asarray(grid)
        grid_value = np.asarray(grid_value)
        print(grid.shape,grid_value.shape)
        print(grid[10,10,10,1],grid_value[10,10,10])
        bd = np.array([bounding_box_min_x,bounding_box_min_y,bounding_box_min_z])
        bd_grid = np.array([grid_res_x,grid_res_y,grid_res_z])
        return grid, grid_value, bd, bd_grid, voxel_size
def safe_norm_np(x, epsilon=1e-12, axis=1):
    return np.sqrt(np.sum(x*x, axis=axis) + epsilon)

def safe_norm(x, epsilon=1e-12, axis=None):
  return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)

def boundingbox(x,y,z):
    return min(x),max(x),min(y),max(y),min(z),max(z)

def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')     # 必须先写入，然后利用write()在头部插入ply header
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)
    
def distance_matrix(array1, array2):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
            , it's size: (num_point, num_point)
    """
    num_point, num_features = array1.shape
    expanded_array1 = tf.tile(array1, (num_point, 1))
    expanded_array2 = tf.reshape(
            tf.tile(tf.expand_dims(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = tf.norm(expanded_array1-expanded_array2, axis=1)
    distances = tf.reshape(distances, (num_point, num_point))
    return distances

def av_dist(array1, array2):
    """
    arguments:
        array1, array2: both size: (num_points, num_feature)
    returns:
        distances: size: (1,)
    """
    distances = distance_matrix(array1, array2)
    distances = tf.reduce_min(distances, axis=1)
    distances = tf.reduce_mean(distances)
    return distances

def av_dist_sum(arrays):
    """
    arguments:
        arrays: array1, array2
    returns:
        sum of av_dist(array1, array2) and av_dist(array2, array1)
    """
    array1, array2 = arrays
    av_dist1 = av_dist(array1, array2)
    av_dist2 = av_dist(array2, array1)
    return av_dist1+av_dist2

def chamfer_distance_tf(array1, array2):
    batch_size, num_point, num_features = array1.shape
    dist = tf.reduce_mean(
               tf.map_fn(av_dist_sum, elems=(array1, array2), dtype=tf.float32)
           )
    return dist

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

def near_point_idx(array1, array2, num_point1,num_point2, num_features = 3):
    array1 = tf.reshape(array1,[-1,3])
    array2 = tf.reshape(array2,[-1,3])
    #num_point, num_features = array1.shape
    expanded_array1 = tf.tile(array1, (num_point2, 1))
    expanded_array2 = tf.reshape(
            tf.tile(tf.expand_dims(array2, 1), 
                    (1, num_point1, 1)),
            (-1, num_features))
    distances = tf.norm(expanded_array1-expanded_array2, axis=1)
    distances = tf.reshape(distances, (num_point2, num_point1))
    dis_idx = tf.argmin(distances, axis=1)
    return dis_idx

feature = tf.placeholder(tf.float32, shape=[BS,None,512])
points_target = tf.placeholder(tf.float32, shape=[BS,None,3])
input_points_3d = tf.placeholder(tf.float32, shape=[BS, None,3])
normal_gt = tf.placeholder(tf.float32, shape=[BS,None,3])
points_target_num = tf.placeholder(tf.int32, shape=[1,1])
points_input_num = tf.placeholder(tf.int32, shape=[1,1])
points_cd = tf.placeholder(tf.float32, shape=[BS,None,3])

#one_hot_f = tf.nn.relu(tf.layers.dense(one_hot,128))
#feature_f = tf.nn.relu(tf.layers.dense(feature,128))
#one_hot_f = tf.nn.softplus(tf.layers.dense(one_hot,128))
#feature_f = tf.nn.softplus(tf.layers.dense(feature,128))
#input_image = tf.expand_dims(input_points_2d,-1)
#net = tf.nn.softplus(tf.layers.conv1d(input_points_2d, 512, 1))
#print('net:',net)
#net = tf.concat([net,one_hot_f,feature_f],2)
#print('net:',net)
##net = tf.nn.softplus(tf.layers.conv1d(net, 2500, 1))
#print('net:',net)
##net = tf.nn.softplus(tf.layers.conv1d(net, 1250, 1))
#net = tf.nn.softplus(tf.layers.conv1d(net, 512, 1))
#net = tf.nn.softplus(tf.layers.conv1d(net, 512, 1))
##print('net:',net)
#net = tf.layers.conv1d(net, 3, 1)
#print('net:',net)
##output_points = net
#output_points = tf.reshape(net,[BS,POINT_NUM,3])


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

near_idx = near_point_idx(points_target,input_points_3d,points_target_num[0,0],points_input_num[0,0])
print('near_idx',near_idx)

point_target_near = tf.gather(points_target,axis=1,indices=near_idx)
print('point_target_near',point_target_near)
g_points = input_points_3d - sdf * grad_norm

cd_gt = chamfer_distance_tf_None(points_cd, g_points)
#loss = tf.losses.huber_loss(point_target_near, g_points)
loss = chamfer_distance_tf_None(point_target_near, g_points)



t_vars = tf.trainable_variables()
optim = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9)
loss_grads_and_vars = optim.compute_gradients(loss, var_list=t_vars)
loss_optim = optim.apply_gradients(loss_grads_and_vars)


config = tf.ConfigProto(allow_soft_placement=False) 
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
saver_restore = tf.train.Saver(var_list=t_vars)
saver = tf.train.Saver(max_to_keep=2000000)

mm = 0
pointclouds = []
samples = []
normals_gt = []

files = []

f = open(test_list,'r')
for index,line in enumerate(f):
    files.append(line.strip().split('/')[1])
f.close()


for file in files:
    print(file)
    mm = mm + 1
    if(mm>SHAPE_NUM):
        break
    print(INPUT_DIR + file)
    data = np.load(INPUT_DIR + file + '/pointcloud.npz')

    pointcloud = data['points'].reshape(1,-1,3)
    normal = data['normals'].reshape(1,-1,3)
    pointcloud = pointcloud.reshape(1,-1,3)
    pointclouds.append(pointcloud)
    normals_gt.append(normal)
    #samples.append(data['sample'])
    
SHAPE_NUM = len(files)
print('SHAPE_NUM:',SHAPE_NUM)   

pointclouds = np.asarray(pointclouds).reshape(-1,POINT_NUM_GT,3)
normals_gt = np.asarray(normals_gt).reshape(-1,POINT_NUM_GT,3)
#samples = np.asarray(samples)
#samples = samples.reshape(SHAPE_NUM,-1,POINT_NUM,3)
print(pointclouds.shape,normals_gt.shape)



with tf.Session(config=config) as sess:
    if(TRAIN):
        print('train start')
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        
        POINT_NUM_GT_bs = np.array(POINT_NUM_GT).reshape(1,1)
        points_input_num_bs = np.array(POINT_NUM).reshape(1,1)
        for i in range(10000):
            epoch_index = np.random.choice(SHAPE_NUM, SHAPE_NUM, replace = False)
            
            for epoch in epoch_index:
                rt = random.randint(0,samples.shape[1]-1)
                input_points_2d_bs = samples[epoch,rt,:,:].reshape(BS, POINT_NUM, 3)
                point_gt = pointclouds[epoch,:,:].reshape(1,-1,3)
                feature_bs_t = feature_bs[epoch,:,:].reshape(1,-1,SHAPE_NUM)
                sess.run([loss_optim],feed_dict={input_points_3d:input_points_2d_bs,points_target:point_gt,feature:feature_bs_t,points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs})
            if(i%100 == 0):
                loss_c,output_points_c,point_target_near_c,near_idx_c = sess.run([loss,g_points,point_target_near,near_idx],feed_dict={input_points_3d:input_points_2d_bs,
                                                                          points_target:point_gt,feature:feature_bs_t,points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs})
                print('i:',i,'epoch:',epoch,'loss:',loss_c)
                
            if(i%5000 == 0):
                print('save model')
                #print(p_points_bs)
                saver.save(sess, os.path.join(OUTPUT_DIR, "model"), global_step=i+1)
                #print(near_idx_c)
    #            with open('./near_{}.txt'.format(i),'w') as f:
    #                for j in range(point_target_near_c.shape[1]):
    #                    f.write(str(point_target_near_c[0,j,0])+';')
    #                    f.write(str(point_target_near_c[0,j,1])+';')
    #                    f.write(str(point_target_near_c[0,j,2])+';')  
    #                    f.write(str(255)+';')
    #                    f.write(str(0)+';')
    #                    f.write(str(0)+';\n')  
    #                for j in range(point_target_near_c.shape[1]):
    #                    f.write(str(input_points_2d_bs[0,j,0])+';')
    #                    f.write(str(input_points_2d_bs[0,j,1])+';')
    #                    f.write(str(input_points_2d_bs[0,j,2])+';')  
    #                    f.write(str(0)+';')
    #                    f.write(str(255)+';')
    #                    f.write(str(0)+';\n')  
        
        end_time = time.time()
        print('run_time:',end_time-start_time)
    else:
        
        print('feature_test:',feature_test.shape)
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
            with open(OUTPUT_DIR + 'gt_' + files[epoch] + '.txt','w') as f:
                for i in range(pointclouds.shape[1]):
                    x = pointclouds[epoch,i,0]
                    y = pointclouds[epoch,i,1]
                    z = pointclouds[epoch,i,2]
                    f.write(str(x)+';')
                    f.write(str(y)+';')
                    f.write(str(z)+';\n')  
#            input_points_2d_bs_t = samples[epoch,0,:,:].reshape(BS, POINT_NUM, 3)
#            point_gt = pointclouds[epoch,:,:].reshape(1,-1,3)
#            feature_bs_t = feature_bs[epoch,:,:].reshape(1,-1,SHAPE_NUM)
#            loss_c,output_points_c,point_target_near_c,near_idx_c = sess.run([loss,g_points,point_target_near,near_idx],feed_dict={input_points_3d:input_points_2d_bs_t,
#                                                                      points_target:point_gt,feature:feature_bs_t,points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs})
#            with open(OUTPUT_DIR + 'test_{}.txt'.format(epoch),'w') as f:
#                for i in range(output_points_c.shape[1]):
#                    f.write(str(output_points_c[0,i,0])+';')
#                    f.write(str(output_points_c[0,i,1])+';')
#                    f.write(str(output_points_c[0,i,2])+';')  
#                    f.write(str(255)+';')
#                    f.write(str(0)+';')
#                    f.write(str(0)+';\n') 
#                for i in range(input_points_2d_bs.shape[1]):
#                    f.write(str(input_points_2d_bs[0,i,0])+';')
#                    f.write(str(input_points_2d_bs[0,i,1])+';')
#                    f.write(str(input_points_2d_bs[0,i,2])+';')  
#                    f.write(str(0)+';')
#                    f.write(str(255)+';')
#                    f.write(str(0)+';\n') 
#                for i in range(point_target_near_c.shape[1]):
#                    f.write(str(point_target_near_c[0,i,0])+';')
#                    f.write(str(point_target_near_c[0,i,1])+';')
#                    f.write(str(point_target_near_c[0,i,2])+';')  
#                    f.write(str(255)+';')
#                    f.write(str(255)+';')
#                    f.write(str(0)+';\n') 
           
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
                    #print('vertices:',vertices.shape,triangles.shape)
        #            net_c = feature_test[epoch*24,:]
        #            net_c = net_c.reshape((-1))
        #            net_c = np.expand_dims(net_c,0).repeat(vertices.shape[0],axis=0)
        #            net_c = net_c.reshape((1,vertices.shape[0], 512))
        #            feature_bs_tt = net_c
        #            grad_norm_c = sess.run([grad_norm],feed_dict={input_points_3d:vertices.reshape(1,-1,3),feature:feature_bs_tt.reshape(1,-1,512)})
        #            #print('grad_norm',grad_norm_c)
        #            grad_norm_c = np.asarray(grad_norm_c)
        #            print(grad_norm_c)
        #            grad_norm_c = -grad_norm_c.reshape(-1,3)
        #            mesh = trimesh.Trimesh(vertices, triangles,
        #                           vertex_normals=grad_norm_c,
        #                           process=False)
        #            mesh.export(OUTPUT_DIR +  '/occ_{}.off'.format(epoch))
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
                    
        #            net_c = feature_test[epoch*24,:]
        #            net_c = net_c.reshape((-1))
        #            net_c = np.expand_dims(net_c,0).repeat(vertices.shape[0],axis=0)
        #            net_c = net_c.reshape((1,vertices.shape[0], 512))
        #            feature_bs_tt = net_c
        #            print('feature_bs_tt:',feature_bs_tt.shape)
        #            grad_norm_c = sess.run([grad_norm],feed_dict={input_points_3d:vertices.reshape(1,-1,3),feature:feature_bs_tt.reshape(1,-1,512)})
        #            #print('grad_norm',grad_norm_c)
        #            grad_norm_c = np.asarray(grad_norm_c)
        #            print(grad_norm_c.shape)
        #            grad_norm_c = grad_norm_c.reshape(-1,3)
                    
                    
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
    