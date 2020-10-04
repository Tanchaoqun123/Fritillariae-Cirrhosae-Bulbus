import os
from frcnn2 import FRCNN2
import csv
from PIL import Image
import sys
FIELDNAMES = ['image_id','image_h','image_w', 'num_boxes','boxes', 'features']
def generate_tsv(outfile, imagePath,savepath):
    with open(outfile, 'a+') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)
        count = 0
        images = os.listdir(imagePath)
        for name in images:
            value = name.split('.')
            name_path = imagePath + name
            image = Image.open(name_path)
            image_id=value[0]
            writer.writerow(frcnn2.detect_image(image_id,image,savepath+name))
            count=count+1
            print(count)


if __name__ == '__main__':
    frcnn2 = FRCNN2()
    savepath='D:/test/faster-rcnn-pytorch-master/predict_image2/'
    outfile='./tsv/feature.tsv'
    imagePath='D:/test/faster-rcnn-pytorch-master/VOCdevkit/VOC2007/JPEGImages/'
    generate_tsv(outfile,imagePath,savepath)