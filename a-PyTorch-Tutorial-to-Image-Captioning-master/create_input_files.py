from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='./result2.json',
                       image_folder='D:/SENet-Tensorflow-master/data_/',
                       captions_per_image=1,
                       min_word_freq=0,
                       output_folder='./media/ssd/caption_data2/',
                       max_len=30)
