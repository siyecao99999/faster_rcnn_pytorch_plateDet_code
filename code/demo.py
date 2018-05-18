import cv2
import os
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer

import xml.etree.ElementTree as ET

def image_test(net, image_file, anno_file):
    tree = ET.parse(anno_file)
    size = tree.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)

    objs = tree.findall('object')
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 4), dtype=np.int32)
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        cx = int(bbox.find('cx').text)
        cy = int(bbox.find('cy').text)
        wid = int(bbox.find('wid').text)
        hei = int(bbox.find('hei').text)
        theta = float(bbox.find('theta').text)
        #boxes[ix, :] = [cx, cy, wid, hei, theta]

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
        xmax = min(img_w-1, xymax[0])
        ymax = min(img_h-1, xymax[1])
        boxes[ix, :] = [xmin, ymin, xmax, ymax]


    image = cv2.imread(image_file)

    t = Timer()
    t.tic()
    # image = np.zeros(shape=[600, 800, 3], dtype=np.uint8) + 255
    dets, scores, classes = net.detect(image, 0.7)
    runtime = t.toc()
    print('total spend: {}s'.format(runtime))

    im2show = np.copy(image)
    
    for box in boxes:
        box = tuple(int(x) for x in box)
        cv2.rectangle(im2show, box[0:2], box[2:4], (0, 0, 255), 2)


    for i, det in enumerate(dets):
        det = tuple(int(x) for x in det)
        cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
        cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
    im_name = os.path.basename(image_file)
    print(os.path.join('demo/det_results', im_name))
    cv2.imwrite(os.path.join('demo/det_results', im_name), im2show)
    #cv2.imshow('demo', im2show)
    #cv2.waitKey(0)


def folder_test(net, folder):
    txt_file = folder + 'JPEGImages/file_name.txt'

    with open(txt_file) as f:
        for line in f:
            img_path = folder + 'JPEGImages/' + line.strip('\n') +'.JPG'
            anno_path = folder + 'Annotations/' + line.strip('\n') +'.xml'
            image_test(net, img_path, anno_path)


if __name__ == '__main__':
    model_file = 'models/saved_model3/faster_rcnn_100000.h5'
    detector = FasterRCNN()
    network.load_net(model_file, detector)
    detector.cuda()
    detector.eval()
    print('load model successfully!')
    #image_file = 'demo/000001.JPG'
    #image_test(detector, image_file, None)

    folder = '/data/jmtian/PlateData/PVW_WRM_CUT/'
    folder_test(detector, folder)
