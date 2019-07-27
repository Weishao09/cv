import numpy as np
import random
import math

#Assignment2 Logistic Regression in Python 
#Hypothesis: y = 1/(1+exp(-w*x-b))
# inference, test, predict, same thing. Run model after training
def inference(w, b, x):
    v = -np.multiply(w,x)-b
    pred_y = 1/(1+np.exp(v))
    return pred_y

def eval_loss(w, b, x_list, gt_y_list):
    avg_loss = 0.0
    # loss function
    log_p = np.log(inference(w, b, x_list))
    log_m = np.log(1 - inference(w, b, x_list))
    avg_loss = -(np.multiply(gt_y_list,log_p)) - (np.multiply((1-gt_y_list),log_m)) 
    avg_loss = np.average(avg_loss)
    return avg_loss
 
def gradient(pred_y, gt_y, x):
    diff = np.array(pred_y) - np.array(gt_y)
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
lr = 0.01
max_iter = 1000
train(x_list,y_list,50,lr,max_iter)

#结果是展示w，b，loss的1000次梯度下降对应的值
#lr取0.01 结果基本类似于
#w:16117.647135715682, b:342.467106053017
#loss is -inf
#lr取0.001 loss的值是nan 不知道是哪里不对了