import json
f = open("caption_data.txt","r",errors='ignore')
lines = f.readlines()
p=0
for line in lines:
    p=p+1
    print(p)