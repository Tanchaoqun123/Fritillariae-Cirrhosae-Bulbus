import os
import json
key=['tokens']
d=['该','炉贝','表面','浅棕黄色',"，","具","棕色","斑点","，","外层鳞叶","2瓣","，","大小相近","，","顶部","开裂","略尖","，","基部","稍尖"]
value=[d]
dictionary = dict(zip(key, value))
j = json.dumps(dictionary)
print(j)
v='['+j+']'
print(v)
keys = ['filepath', 'filename', 'split']
values = ["train", "COCO_val2014_000000391895.jpg", "train"]
dictionarys = dict(zip(keys, values))
js = json.dumps(dictionarys)
print(js)
kkk=js[:-1]
print(kkk)
k=kkk+",\"sentences\":"+v+"}"
print(k)
f = open('result.json','w')
f.write(k)


