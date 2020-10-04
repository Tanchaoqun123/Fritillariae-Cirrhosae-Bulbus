import cv2
import numpy as np
import colorsys
import os
import torch
import time
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from utils.utils import loc2bbox, nms, DecodeBox
from nets.frcnn import FasterRCNN
from nets.frcnn_training import get_new_img_size
from PIL import Image, ImageFont, ImageDraw
import copy
import math
import base64


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
# --------------------------------------------#
class FRCNN2(object):
    _defaults = {
        "model_path": 'logs/Epoch49-Total_Loss0.3838-Val_Loss0.4336.pth',
        "classes_path": 'model_data/voc_classes.txt',
        "confidence": 0.6,
        "backbone": "resnet50"
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化faster RCNN
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()
        self.mean = torch.Tensor([0, 0, 0, 0]).cuda().repeat(self.num_classes + 1)[None]
        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).cuda().repeat(self.num_classes + 1)[None]

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        # 计算总的种类
        self.num_classes = len(self.class_names)

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        self.model = FasterRCNN(self.num_classes, "predict", backbone=self.backbone).cuda()
        self.model.load_state_dict(torch.load(self.model_path))
        cudnn.benchmark = True

        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image_id,image,savepath):
        image_shape = np.array(np.shape(image)[0:2])
        old_width = image_shape[1]
        old_height = image_shape[0]
        old_image = copy.deepcopy(image)
        width, height = get_new_img_size(old_width, old_height)
        image = image.resize([width, height])
        photo = np.array(image, dtype=np.float32) / 255
        photo = np.transpose(photo, (2, 0, 1))
        with torch.no_grad():
            images = []
            images.append(photo)
            images = np.asarray(images)
            images = torch.from_numpy(images).cuda()

            roi_cls_locs, roi_scores, rois, roi_indices, feature = self.model(images)
            decodebox = DecodeBox(self.std, self.mean, self.num_classes)
            outputs = decodebox.forward(roi_cls_locs, roi_scores, rois, feature, height=height, width=width,
                                              score_thresh=self.confidence)
            if len(outputs) == 0:
                return old_image
            if np.size(outputs, 0)>4:
                outputs = outputs[np.argsort(outputs[:, 4])]
                outputs = outputs[-4:, :]
            bbox = outputs[:, :4]
            conf = outputs[:, 4]
            label = outputs[:, 5]
            f = outputs[:, 6:]
            f_size=np.size(f, 0)
            if f_size<4:
                k=4-f_size
                cc=0
                for lab in label:
                    if (lab ==1) or (lab==3) or (lab==5):
                        conbin_f = f[cc,:]
                        conbin_f=conbin_f.reshape(1,2048)
                        for num in range(0, k):
                            f = np.append(f, conbin_f, axis=0)
                        break
                    cc=cc+1
            bbox[:, 0::2] = (bbox[:, 0::2]) / width * old_width
            bbox[:, 1::2] = (bbox[:, 1::2]) / height * old_height
            bbox = np.array(bbox, np.int32)
            f=np.array(f, np.float32).reshape((4,2048))
            print(np.size(f))
            # image = old_image
            # thickness = (np.shape(old_image)[0] + np.shape(old_image)[1]) // old_width * 2
            # font = ImageFont.truetype(font='model_data/simhei.ttf',
            #                           size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
            #
            # for i, c in enumerate(label):
            #     predicted_class = self.class_names[int(c)]
            #     score = conf[i]
            #
            #     left, top, right, bottom = bbox[i]
            #     top = top - 5
            #     left = left - 5
            #     bottom = bottom + 5
            #     right = right + 5
            #
            #     top = max(0, np.floor(top + 0.5).astype('int32'))
            #     left = max(0, np.floor(left + 0.5).astype('int32'))
            #     bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            #     right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))
            #
            #     # 画框框
            #     label = '{} {:.2f}'.format(predicted_class, score)
            #     draw = ImageDraw.Draw(image)
            #     label_size = draw.textsize(label, font)
            #     label = label.encode('utf-8')
            #     print(label)
            #
            #     if top - label_size[1] >= 0:
            #         text_origin = np.array([left, top - label_size[1]])
            #     else:
            #         text_origin = np.array([left, top + 1])
            #
            #     for i in range(thickness):
            #         draw.rectangle(
            #             [left + i, top + i, right - i, bottom - i],
            #             outline=self.colors[int(c)])
            #     draw.rectangle(
            #         [tuple(text_origin), tuple(text_origin + label_size)],
            #         fill=self.colors[int(c)])
            #     draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            #     del draw
            #
            # image.save(savepath)
        if np.size(f, 0)!=4:
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        return {
            'image_id': image_id,
            'image_h': height,
            'image_w': width,
            'num_boxes': np.size(bbox, 0),
            'boxes': base64.b64encode(bbox),
            'features': base64.b64encode(f)
        }
