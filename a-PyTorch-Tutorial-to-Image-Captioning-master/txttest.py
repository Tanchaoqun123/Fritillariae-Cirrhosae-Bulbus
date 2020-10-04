import json
f = open("caption_data.txt","r",encoding='utf8')
lines = f.readlines()
p=""
for line in lines:
    print(line)
    s=line[:-1].split(' ')
    filename=s[0]
    caption=s[1].split('/')
    va=[]
    for c in caption:
        va.append(c)

    key = ['tokens']
    value = [va]
    dictionary = dict(zip(key, value))
    j = json.dumps(dictionary)
    v = '[' + j + ']'
    keys = ['filepath', 'filename', 'split']
    values = ["train", filename, "train"]
    dictionarys = dict(zip(keys, values))
    js = json.dumps(dictionarys)
    print(js)
    kkk = js[:-1]
    print(kkk)
    k = kkk + ",\"sentences\":" + v + "}"
    print(k)
    p=p+k+','
f = open('result2.json', 'w')
f.write(p)