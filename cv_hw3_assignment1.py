import numpy as np
import random

#Assignment1 Linear Regression in Python 

# inference, test, predict, same thing. Run model after training
def inference(w, b, x):        
    pred_y = np.multiply(w,x)+b
    return pred_y

def eval_loss(w, b, x_list, gt_y_list):
    avg_loss = 0.0
    # loss function
    avg_loss = 0.5 * (np.multiply(w, x_list) + b - gt_y_list) ** 2    
    avg_loss = np.average(avg_loss)
    return avg_loss
 
def gradient(pred_y, gt_y, x):
    diff = pred_y - gt_y
    dw = np.multiply(diff,x)	
    db = diff
    return dw, db

def cal_step_gradient(batch_x_list, batch_gt_y_list, w, b, lr):
	avg_dw, avg_db = 0, 0
	pred_y = inference(w, b, batch_x_list)	# get label data
	dw, db = gradient(pred_y, batch_gt_y_list, batch_x_list)
	avg_dw = np.average(dw)
	avg_db = np.average(db)
	w -= lr * avg_dw
	b -= lr * avg_db
	return w, b

def train(x_list, gt_y_list, batch_size, lr, max_iter):
    w = 0
    b = 0
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(x_list), batch_size)
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [gt_y_list[j] for j in batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0}, b:{1}'.format(w, b))
        print('loss is {0}'.format(eval_loss(w, b, x_list, gt_y_list)))

def gen_sample_data():
	w = random.randint(0, 10) + random.random()		# for noise random.random[0, 1)
	b = random.randint(0, 5) + random.random()
	num_samples = 100
	x_list = [random.randint(0, 100) * random.random() for i in range(num_samples)]
	y_list = np.multiply(x_list,w) + b + [random.random() * random.randint(-1, 1) for i in range(num_samples)]
	return x_list, y_list, w, b

x_list, y_list, w, b = gen_sample_data()
lr = 0.001
max_iter = 10000
train(x_list,y_list,50,lr,max_iter)

#结果是展示w，b，loss的10000次梯度下降对应的值
#lr取0.001 结果基本类似于
#w:7.211457841066508, b:1.0867379971731062
#loss is 0.2808228702398605