import os
import csv
from PIL import Image
imagePath='D:/SENet-Tensorflow-master/data_/train/'
images = os.listdir(imagePath)
count=0
for name in images:
    line=name+'\n'
    with open('caption_data.txt', 'a') as f:
        f.write(line)
    count=count+1
    print(count)