import os
import numpy as np
import json

phase = "val"
DATA_FILE_PATH = f'../../../datasets/kag/nfl.{phase}'
OUT_PATH = '../../../datasets/kag/annotations/'
IMAGE_PATH = '../../../datasets/kag/dataset/images/'
LABEL_PATH = '../../../datasets/kag/dataset/labels/'
def load_paths(data_path):
    with open(data_path, 'r') as file:
        files = file.readlines()
        files = [x.replace('\n', '') for x in files]
        files = list(filter(lambda x: len(x) > 0, files))
    img_files = [os.path.join(IMAGE_PATH, x) for x in files]
    label_files = [os.path.join(LABEL_PATH, x.replace('.jpg', '.txt')) for x in files]
    return img_files, label_files                    

if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)

    out_path = OUT_PATH + f'{phase}.json'
    out = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'helmet'}]}
    img_paths, label_paths = load_paths(DATA_FILE_PATH)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    for img_path, label_path in zip(img_paths, label_paths):
        image_cnt += 1
        # im = Image.open(img_path)
        image_info = {'file_name': img_path, 
                        'id': image_cnt,
                        'height': 720,
                        'width': 1080}
        out['images'].append(image_info)
        # Load labels
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = image_info['width'] * (labels0[:, 2] - labels0[:, 4] / 2)
            labels[:, 3] = image_info['height'] * (labels0[:, 3] - labels0[:, 5] / 2)
            labels[:, 4] = image_info['width'] * labels0[:, 4]
            labels[:, 5] = image_info['height'] * labels0[:, 5]
        else:
            labels = np.array([])
        for i in range(len(labels)):
            ann_cnt += 1
            fbox = labels[i, 2:6].tolist()
            ann = {'id': ann_cnt,
                    'category_id': 1,
                    'image_id': image_cnt,
                    'track_id': -1,
                    'bbox': fbox,
                    'area': fbox[2] * fbox[3],
                    'iscrowd': 0}
            out['annotations'].append(ann)
    print('loaded {} for {} images and {} samples'.format(phase, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))