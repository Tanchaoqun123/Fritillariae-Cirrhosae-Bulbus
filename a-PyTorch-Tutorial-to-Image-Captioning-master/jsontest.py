import json
with open('./media/ssd/caption_data2/WORDMAP_coco_1_cap_per_img_0_min_word_freq.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
    print('这是文件中的json数据：',json_data)