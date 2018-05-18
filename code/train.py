import os
import torch
import numpy as np
import cv2
from datetime import datetime

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)



# hyper-parameters
# ------------
imdb_name = 'PVW_WRM_CUT_trainval'
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
pretrained_model = 'models/pretrained_model/VGG_imagenet_soft.npy'
pretrained_fcn = 'models/pretrained_model/fcn_soft.pth'

output_dir = 'models/saved_model3'

start_step = 0
end_step = 100000
lr_decay_steps = {60000, 80000}
lr_decay = 1./10

rand_seed = 1024
_DEBUG = False
use_tensorboard = False
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load data
imdb = get_imdb(imdb_name)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb
data_layer = RoIDataLayer(roidb, imdb.num_classes)

# load net
net = FasterRCNN(classes=imdb.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.01)
network.load_pretrained_npy(net, pretrained_model)
#network.load_pretrained_fcn(net, pretrained_fcn)

net.cuda()
net.train()

params = list(net.parameters())
optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:
        exp_name = datetime.now().strftime('vgg16_%m-%d_%H-%M')
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()
for step in range(start_step, end_step+1):

    # get one batch
    blobs = data_layer.forward()
    im_data = blobs['data']
    im_info = blobs['im_info']
    gt_boxes_6d = blobs['gt_boxes']


    """
    img = im_data[0] + cfg.PIXEL_MEANS
    img = img.astype(np.uint8)
    for idx in xrange(gt_boxes_6d.shape[0]):
        cx = gt_boxes_6d[idx][0]
        cy = gt_boxes_6d[idx][1]
        wid = gt_boxes_6d[idx][2]
        hei = gt_boxes_6d[idx][3]
        theta = gt_boxes_6d[idx][4]
        if theta >0:
            rect = ((cx, cy), (wid, hei), -theta)
        else:
            rect = ((cx, cy), (hei, wid), -90-theta)
        pts = cv2.boxPoints(rect)
        pts = np.array(pts, np.int32)
        img = cv2.line(img, (pts[0][0],pts[0][1]),(pts[1][0],pts[1][1]), (0,0,255),2)
        img = cv2.line(img, (pts[1][0],pts[1][1]),(pts[2][0],pts[2][1]), (0,0,255),2)
        img = cv2.line(img, (pts[2][0],pts[2][1]),(pts[3][0],pts[3][1]), (0,0,255),2)
        img = cv2.line(img, (pts[3][0],pts[3][1]),(pts[0][0],pts[0][1]), (0,0,255),2)
        cv2.imwrite("input/"+blobs['im_name'], img)

    if step > len(data_layer._roidb):
            break

    """


    # convert[cx, cy, w, h, theta, label] to [xmin, ymin, xmax, ymax, label]
    gt_boxes = np.empty((gt_boxes_6d.shape[0], 5), gt_boxes_6d.dtype)
    im_size = (im_info[0][0], im_info[0][1])
    for k in xrange(gt_boxes_6d.shape[0]):
        cx = gt_boxes_6d[k][0]
        cy = gt_boxes_6d[k][1]
        wid = gt_boxes_6d[k][2]
        hei = gt_boxes_6d[k][3]
        theta = gt_boxes_6d[k][4]

        if theta >0:
            rect = ((cx, cy), (wid, hei), -theta)
        else:
            rect = ((cx, cy), (hei, wid), -90-theta)
        pts = cv2.boxPoints(rect)
        pts = np.array(pts, np.int32)
        xymin = np.min(pts, axis=0).tolist()
        xymax = np.max(pts, axis=0).tolist()
        xmin = max(0,xymin[0])
        ymin = max(0,xymin[1])
        xmax = min(im_size[1]-1, xymax[0])
        ymax = min(im_size[0]-1, xymax[1])
        gt_boxes[k] = np.array([xmin, ymin, xmax, ymax, gt_boxes_6d[k][5]])

    # forward
    net(im_data, im_info, gt_boxes)
    loss = net.loss + net.rpn.loss

    if _DEBUG:
        tp += float(net.tp)
        tf += float(net.tf)
        fg += net.fg_cnt
        bg += net.bg_cnt

    train_loss += loss.data[0]
    step_cnt += 1

    # backward
    optimizer.zero_grad()
    loss.backward()
    network.clip_gradient(net, 10.)
    optimizer.step()

    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration

        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
            step, blobs['im_name'], train_loss / step_cnt, fps, 1./fps)
        log_print(log_text, color='green', attrs=['bold'])

        if _DEBUG:
            log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' % (tp/fg*100., tf/bg*100., fg/step_cnt, bg/step_cnt))
            log_print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f' % (
                net.rpn.cross_entropy.data.cpu().numpy()[0], net.rpn.loss_box.data.cpu().numpy()[0],
                net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
            )
        re_cnt = True

    if use_tensorboard and step % log_interval == 0:
        exp.add_scalar_value('train_loss', train_loss / step_cnt, step=step)
        exp.add_scalar_value('learning_rate', lr, step=step)
        if _DEBUG:
            exp.add_scalar_value('true_positive', tp/fg*100., step=step)
            exp.add_scalar_value('true_negative', tf/bg*100., step=step)
            losses = {'rpn_cls': float(net.rpn.cross_entropy.data.cpu().numpy()[0]),
                      'rpn_box': float(net.rpn.loss_box.data.cpu().numpy()[0]),
                      'rcnn_cls': float(net.cross_entropy.data.cpu().numpy()[0]),
                      'rcnn_box': float(net.loss_box.data.cpu().numpy()[0])}
            exp.add_scalar_dict(losses, step=step)

    if (step % 10000 == 0) and step > 0:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))
    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False
    
