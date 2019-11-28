# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 16:10:25 2019
@author: lalit.h.suthar
"""

from flask import Flask, render_template, redirect, url_for, request, jsonify, make_response, Response
from werkzeug import secure_filename
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import random
import os
import json

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]
label_list = [i for i in COCO_INSTANCE_CATEGORY_NAMES[1:] if i != 'N/A' ]

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/img')
def img():
    return render_template('img.html',
                           label=None,
                           labels=label_list,
                           msg=None,
                           ifile=None,
                           ofile=None
                           )


@app.route('/imgview', methods=['POST','GET'])
def imgview():
    msg = ''
    f = None
    label=''
    input_file_path=''
    out_file_path=''

    if request.method == 'POST':
        f = request.files['imgfileupload']
        label = request.form.get('label_name')
        #print(label)
        #print(secure_filename(f.filename))
        input_file_path = os.path.join("static/data/", secure_filename(f.filename))

        if os.path.exists(input_file_path):
            os.remove(input_file_path)

        f.save(input_file_path)

        try:
            out_file_path = instance_segmentation_api(label=label, img_path=input_file_path)
            msg = 'File ' + str(f.filename) + ' Uploaded and Processed Successfully !'
            return render_template('img.html',
                                   label=label,
                                   msg=msg,
                                   labels=label_list,
                                   ifile=input_file_path,
                                   ofile=out_file_path)
        except Exception as e:
            msg = ' Try other Image, ML processing failed : ' + str(e)
            out_file_path=''
            render_template('img.html', msg=msg, labels=label_list, out_file_path=out_file_path)
    else:
        msg = 'Upload Failed or Request Failed !'
        return render_template('img.html', msg=msg, labels=label_list)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/data', methods=['POST','GET'])
def data():
    os.walk()
    return render_template()


def random_colour_masks(image):
    """
    random_colour_masks
    parameters:
      - image - predicted masks
    method:
      - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def get_prediction(img_path, threshold, label):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0

    """
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    # print(pred[0]['person'])
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    indexes = [i for i in range(len(pred_class)) if pred_class[i] == label]
    masks = [masks[i] for i in indexes]
    pred_boxes = [pred_boxes[i] for i in indexes]
    pred_class = [i for i in pred_class if i == label]

    data = {}
    data['masks'] = str(masks) # np.array(masks).tolist() #
    data['coordinates'] = str(pred_boxes)
    data['labels'] = pred_class

    json_str = json.dumps(data, indent=4, sort_keys=True)
    with open(str(img_path.split('.')[0])+"_"+label+"_out.json", 'w') as outfile:
        outfile.write(json_str)

    return masks, pred_boxes, pred_class


def instance_segmentation_api(label, img_path, threshold=0.85, rect_th=2, text_size=1, text_th=2):
    """
    instance_segmentation_api
    parameters:
      - img_path - path to input image
    method:
      - prediction is obtained by get_prediction
      - each mask is given random color
      - final output is displayed
    """
    output_img = str(img_path.split('.')[0])+"_"+label+"_out.jpg"

    if os.path.exists(output_img):
        os.remove(output_img)

    masks, boxes, pred_cls = get_prediction(img_path, threshold, label)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 1, 0)
        #cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        #cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_ITALIC, text_size, (255,0,0),thickness=text_th)

    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(output_img, bbox_inches='tight', pad_inches=0, transparent=False)
    #plt.show()

    #output_dict = {'masks': np.array(masks).tolist()}
    #{'pred_boxes': boxes, 'pred_class': pred_cls}
    #output_json = json.dumps(output_dict)
    #'masks': np.array(masks).tolist(),
    #with open(output_img+'.json', 'w') as outfile:
    #    json.dump(output_dict, outfile, ensure_ascii=False)

    return output_img


if __name__ == '__main__':
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    app.run(debug=True)