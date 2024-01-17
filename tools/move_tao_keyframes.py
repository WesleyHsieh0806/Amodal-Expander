import argparse
import os
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt')
    parser.add_argument('--img_dir')
    parser.add_argument('--out_dir')
    args = parser.parse_args()

    data = json.load(open(args.gt, 'r'))
    keyframes = set()
    for i, image in enumerate(data['images']):
        print("[{}/{}]".format(i, len(data['images'])), end='\r')
        img_path = args.img_dir + '/' + image['file_name']
        if not os.path.exists(img_path):
            print('No exists!', img_path)
        target_path = args.out_dir + '/' + image['file_name']
        folder_name = target_path[:target_path.rfind('/')]
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        cmd = 'cp {} {}'.format(img_path, target_path)
        keyframes.add(target_path)
        os.system(cmd)
    print()


