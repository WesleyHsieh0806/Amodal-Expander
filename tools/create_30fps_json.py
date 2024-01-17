import json
import argparse
import os 
from collections import defaultdict

from pathlib import Path
from tqdm import tqdm 
from natsort import natsorted
from tao.utils import fs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True, help='The path of the original annotation file')
    parser.add_argument("--images-dir", type=Path, required=True, help='The path to TAO frame dir')
    parser.add_argument("--fps", type=int, required=True, help='fps')
    
    return parser.parse_args()

def obtain_video_related_info(video_id, video_to_one_fps_images):
    '''
    Input: video_id (int)
    Output:
        width
        height
        license
        _scale_task_id
    '''
    if video_id not in video_to_one_fps_images:
        raise ValueError("video_id {} does not exist".format(video_id))

    image_sample = video_to_one_fps_images[video_id][0]
    return (image_sample['width'], image_sample['height'], image_sample['license'], image_sample['_scale_task_id'])

def extract_30fps_images(dataset, args):
    # build a dict that maps video_id to a list of images
    video_to_one_fps_images = defaultdict(list)
    for image in dataset['images']:
        video_to_one_fps_images[image['video_id']].append(image)
    
    # map file name to original image id
    img_id_old_to_new = dict()
    file_name_to_img_id = {
        image['file_name']: image['id'] for image in dataset['images']
    }  # file_name ('val/YFCC100M/v_25685519b728afd746dfd1b2fe77c/frame0811.jpg')

    cur_id = 0
    new_images = []
    old_images = dataset['images']
    vid_to_video = dict()
    num_annotated = 0
    for video in tqdm(dataset['videos']):
        vname, vid = video['name'], video['id']
        vid_to_video[vid] = video

        # load all images
        frames_dir = args.images_dir / vname
        if not frames_dir.exists():
            logging.warn(f"Could not find images at {frames_dir}")
            return

        frames = natsorted(fs.glob_ext(frames_dir, fs.IMG_EXTENSIONS))

        #######
        # Trim to only show labeled segment.
        #######
        first = next(i for i, f in enumerate(frames)
                    if (os.path.join(vname, f.name) in file_name_to_img_id))
        last = next(i for i, f in reversed(list(enumerate(frames)))
                    if os.path.join(vname, f.name) in file_name_to_img_id)
        
        # print(first, last, vname)

        frames = frames[first:last+1:]
        ann_frame_idxs = [i for i in range(len(frames)) if os.path.join(vname, frames[i].name) in file_name_to_img_id]
        ann_frame_idxs.append(ann_frame_idxs[-1] + 1)  # dummy idx, ensure all annotated images are included
        
        #####
        # load each image and assign it with a new id
        #####
        for i in range(len(ann_frame_idxs) - 1):
            # select 'fps' images between two annotated images 
            start, end = ann_frame_idxs[i], ann_frame_idxs[i + 1]
            step = max((end - start) // args.fps, 1)

            for frame_id, frame in enumerate(frames[start:end:step]):
                file_name = os.path.join(vname, frame.name)
                if file_name in file_name_to_img_id:
                    old_img_id = file_name_to_img_id[file_name]
                    img_id_old_to_new[old_img_id] = cur_id
                
                width, height, license, scale_task_id = obtain_video_related_info(vid, video_to_one_fps_images)
                
                # create new image
                new_image = {
                    'id': cur_id,
                    'frame_id': frame_id,  # to satisfy TET format
                    'video': vname,
                    '_scale_task_id': scale_task_id,
                    'width': width,
                    'height': height,
                    'file_name': file_name,
                    'frame_index': cur_id,
                    'license': license,
                    'video_id': vid,
                    'neg_category_ids': vid_to_video[vid]['neg_category_ids'],
                    'not_exhaustive_category_ids': vid_to_video[vid]['not_exhaustive_category_ids'],
                    'evaluate': file_name in file_name_to_img_id
                }

                num_annotated += int(new_image['evaluate'])
                cur_id += 1
                new_images.append(new_image)
    dataset['images'] = new_images

    img_id_to_vid = {
        image['id']: image['video_id'] for image in dataset['images']
    }

    # update image id in annotations
    new_anns = []
    old_anns = dataset['annotations']
    for ann in dataset['annotations']:
        if ann['image_id'] in img_id_old_to_new:
            ann['image_id'] = img_id_old_to_new[ann['image_id']]
            new_anns.append(ann)

            assert img_id_to_vid[ann['image_id']] == ann['video_id']
    dataset['annotations'] = new_anns
    print("Original # annotations:{}".format(len(old_anns)))
    print("New # annotations:{}".format(len(dataset['annotations'])))

    # Modify the image_id in each annotations
    print("Total number of {} fps images (inside labeled segments): {}".format(args.fps, len(dataset['images'])))
    print("Total number of 1 fps annotated images: {}".format(len(old_images)))
    print("Total number of 1 fps annotated images included in new ann file: {}".format(num_annotated))
    return dataset 

if __name__ == "__main__":
    args = get_args()

    print("Load original annotation file from {}...".format(args.annotations))
    with open(args.annotations, 'r') as f:
        dataset = json.load(f)
    print("Done!")

    dataset = extract_30fps_images(dataset, args)

    new_annotation_file = os.path.splitext(args.annotations)[0] + "_{}fps.json".format(args.fps)
    print("Writing to {}...".format(new_annotation_file))
    with open(new_annotation_file, 'w') as f:
        json.dump(dataset, f)