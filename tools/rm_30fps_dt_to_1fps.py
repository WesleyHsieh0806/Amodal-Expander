import json
import argparse
import os 
from collections import defaultdict

from pathlib import Path
from tqdm import tqdm 

'''
* This script removes the tracking results on non-annotated images. 
The remaining tracking results are those predicted on images annotated at 1 fps
'''
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True, help='The path of the original annotation file')
    parser.add_argument("--track-results", type=Path, required=True, help='The path to 30fps tracking result')
    
    return parser.parse_args()

def remove_results_on_non_labeled_images(dataset, track_results):
    '''
        * track_results : List[prediction]
            prediction is formatted as:
            {
                'image_id': int,
                'video_id':  int,
                'track_id': int,
                'category_id': int,
                'bbox': [x, y, w, h],
                'score': float
            }
    '''
    evaluate_image_or_not = {image['id']: image['evaluate'] for image in dataset['images']}

    new_track_results = []
    images_to_be_evaluated = set()
    print("Original # track results:{}".format(len(track_results)))
    for track_result in track_results:
        if evaluate_image_or_not[track_result['image_id']]:
            new_track_results.append(track_result)
            images_to_be_evaluated.add(track_result['image_id'])
    
    print("# annotated images containing dt results:{}".format(len(images_to_be_evaluated)))
    print("# images containing annotations:{}".format(sum(evaluate_image_or_not.values())))
    print("# track results at 1 fps:{}".format(len(new_track_results)))
    return new_track_results

if __name__ == "__main__":
    args = get_args()

    print("Load original annotation file from {}...".format(args.annotations))
    with open(args.annotations, 'r') as f:
        dataset = json.load(f)
    print("Done!")

    print("Load tracking result file from {}...".format(args.track_results))
    with open(args.track_results, 'r') as f:
        track_results = json.load(f)
    print("Done!")

    new_track_results = remove_results_on_non_labeled_images(dataset, track_results)

    new_track_result_file = os.path.splitext(args.track_results)[0] + "_to_1fps.json" 
    print("Writing to {}...".format(new_track_result_file))
    with open(new_track_result_file, 'w') as f:
        json.dump(new_track_results, f)