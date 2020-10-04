from frcnn import FRCNN
from PIL import Image
import os

def batch_predict(imagePath):
    images = os.listdir(imagePath)
    for name in images:
        name_path = imagePath + name
        image = Image.open(name_path)
        img= frcnn.detect_image(image)
        img.save(path+name)



if __name__ == '__main__':
    frcnn = FRCNN()
    path = 'D:/test/faster-rcnn-pytorch-master/predict_image/'
    frcnn = FRCNN()
    outfile='./tsv/feature.tsv'
    imagePath='D:/test/faster-rcnn-pytorch-master/img/'
    batch_predict(imagePath)