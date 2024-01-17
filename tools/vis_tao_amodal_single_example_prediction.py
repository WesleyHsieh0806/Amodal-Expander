import os
import cv2
import numpy as np
import json
from detectron2.data import detection_utils as utils
from IPython.display import display
from PIL import Image
import json
from collections import defaultdict
annotation_file = '/home/chengyeh/TAO-Amodal-Root/TAO-GTR/datasets/tao/amodal_annotations/validation_with_freeform_amodal_boxes_Aug10_2022_GTR_lvis_v1.json'
prediction_file = '/data3/chengyeh/TAO-Amodal-experiments/GTR/AmodalExpander/TAO-Amodal/ModalMatch/PasteNOcclude/GTR_TAO_Amodal_Expander_PasteNOcclude/iter45000/inference_tao_amodal_val_v1/lvis_instances_results.json'

with open(annotation_file, 'r') as f:
    tao_amodal = json.load(f)

with open(prediction_file, 'r') as f:
    prediction = json.load(f)

modal_prediction_file = '/data3/chengyeh/TAO-Amodal-experiments/GTR/GTR_TAO_Amodal/inference_tao_amodal_val_v1/lvis_instances_results.json'
with open(modal_prediction_file, 'r') as f:
    modal_prediction = json.load(f)

img_id_to_img = {image['id']: image for image in tao_amodal['images']}
vname_to_img_ids = defaultdict(list)
for img in tao_amodal['images']:
    vname_to_img_ids[img['video']].append(img['id'])

img_id_to_prediction = defaultdict(list)
img_id_to_modal_prediction = defaultdict(list)
img_id_to_ann = defaultdict(list)
 
for pred in prediction:
    img_id_to_prediction[pred['image_id']].append(pred)

for pred in modal_prediction:
    img_id_to_modal_prediction[pred['image_id']].append(pred)

for ann in tao_amodal['annotations']:
    img_id_to_ann[ann['image_id']].append(ann)

import numpy as np
import collections
import itertools

def colormap(rgb=False, as_int=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    if as_int:
        color_list = color_list.astype(np.uint8)
    return color_list
color_generator = itertools.cycle(colormap(rgb=True).tolist())
color_map = collections.defaultdict(lambda: next(color_generator))
_BLACK = (0, 0, 0)
_RED = (255, 0, 0)
_BLUE = (0, 0, 255)
_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)

_COLOR1 = tuple(255*x for x in (0.000, 0.447, 0.741))



def vis_image_and_annotations(image, all_annos):
    height, width = image.shape[:2]
    new_image = np.ones([height * 2, width * 2, 3], dtype=np.uint8) * 255
    startx = int(width / 2)
    endx = startx + width
    starty = int(height / 2) 
    endy = starty + height
    new_image[starty: endy, startx: endx, :] = image

    with open('/home/chengyeh/TAO-Amodal-Root/TAO-GTR/datasets/lvis/lvis_v1_train+coco_box.json', 'r') as f:
        lvis = json.load(f)
        id_to_cat_name = {cat['id']: cat['name'] for cat in lvis['categories']}

    for ann in all_annos:
        # if ann['category_id'] != 793 or ann['track_id'] in [1000001, 1000004, 1000007, 1000008]:
        #     continue
        oy, ox = new_image.shape[:2]
        oy, ox = int(oy / 4), int(ox / 4)

        box = ann['bbox']
        box = [box[0]+ox, box[1]+oy, box[2]+box[0]+ox, box[3]+box[1]+oy]
        new_image = vis_bbox(new_image, box, fill_opacity=0.32, fill_color=color_map[ann['track_id']], border_color=_BLACK, thickness=2)
        new_image = vis_class(new_image, box[:2], id_to_cat_name[ann['category_id']])
    

    # Check bounding box, Check Category
    if all_annos:
        new_image = vis_class(new_image, [int(startx + (endx - startx)* 0.4 ), starty // 2], str(all_annos[-1]['image_id']),
                                    bg_color=(255, 255, 255),
                                    text_color=(0, 0, 0),
                                    font_scale=2.5,
                                    thickness=3)
    pil_image = Image.fromarray(new_image)
    return pil_image

def vis_class(image,
              pos,
              class_str,
              font_scale=1.0,
              bg_color=_BLACK,
              text_color=_GRAY,
              thickness=2):
    """Visualizes the class."""
    x, y = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x, y
    back_br = x + txt_w, y + int(1.3 * txt_h)
    # Show text.
    txt_tl = x, y + int(1 * txt_h)
    cv2.rectangle(image, back_tl, back_br, bg_color, -1)
    cv2.putText(image,
                txt,
                txt_tl,
                font,
                font_scale,
                text_color,
                thickness=thickness,
                lineType=cv2.LINE_AA)
    return image

def vis_bbox(image,
             box,
             border_color=_BLACK,
             fill_color=_COLOR1,
             fill_opacity=0.65,
             thickness=2):
    """Visualizes a bounding box."""
    x0, y0, x1, y1 = box
    x1, y1 = int(x1), int(y1)
    x0, y0 = int(x0), int(y0)
    # Draw border
    if fill_opacity > 0 and fill_color is not None:
        with_fill = image.copy()
        with_fill = cv2.rectangle(with_fill, (x0, y0), (x1, y1),
                                  tuple(fill_color), cv2.FILLED)
        image = cv2.addWeighted(with_fill, fill_opacity, image,
                                1 - fill_opacity, 0, image)
        
    image = cv2.rectangle(image, (x0, y0), (x1, y1), tuple(border_color),
                          thickness)
    return image

save_dir = os.path.dirname(os.path.dirname(prediction_file))
display_on_notebook = True
IMAGE_ROOT='/compute/trinity-1-38/chengyeh/TAO/frames'

print(vname_to_img_ids['val/AVA/keUOiCcHtoQ_scene_23_102872-103750'])
# Randomly visualize 100 images
img_ids = np.random.choice(list(img_id_to_img.keys()), size=500)

with open(os.path.join(save_dir, 'random_prediction.html'), 'w') as f:
    f.write('<!DOCTYPE html>\n')
    f.write('<html>\n')
    f.write('<body>\n')
    f.write('<style>\n')
    f.write('table, th, td {\n')
    f.write('border:1px solid black;\n')
    f.write('}\n')
    f.write('</style>\n')
    f.write('<body>\n')
    f.write('<h2>Qualitative Results</h2>\n\n')
    f.write(('<table style="width:100%">\n'
                '<tr>\n'
                '<th>Modal</th>\n'
                '<th>Amodal</th>\n'
                '<th>GT</th>\n'
                '</tr>\n')
                )
    for i,img_id in enumerate(img_ids):
        print("{}/{}".format(i + 1, len(img_ids)), end='\r')
        selected_prediction = [pred for pred in img_id_to_prediction[img_id] if pred['score'] >= 0.5]
        # for pred in selected_prediction:
        #     print(pred)

        img_path = os.path.join(IMAGE_ROOT, img_id_to_img[img_id]['file_name'])
        np_image = utils.read_image(img_path)
        pil_image = vis_image_and_annotations(np_image, selected_prediction)
        amodal_img_path = os.path.join(save_dir, 'predictions', img_id_to_img[img_id]['file_name'].replace('.', '_amodal.'))
        if not os.path.isdir(os.path.dirname(amodal_img_path)):
            os.makedirs(os.path.dirname(amodal_img_path))
        pil_image.save(amodal_img_path)

        # Amodal Predictions

        # Modal Predictions
        new_modal_prediction = [pred for pred in img_id_to_modal_prediction[img_id] if pred['score'] >= 0.5]
        # for pred in new_modal_prediction:
        #     print(pred)
        pil_image = vis_image_and_annotations(np_image, new_modal_prediction)
        pil_image.save(os.path.join(save_dir, 'predictions', img_id_to_img[img_id]['file_name'].replace('.', '_modal.')))


        # GT 
        selected_anns = [pred for pred in img_id_to_ann[img_id]]
        # for ann in selected_anns:
        #     print(ann)
        pil_image = vis_image_and_annotations(np_image, selected_anns)
        pil_image.save(os.path.join(save_dir, 'predictions', img_id_to_img[img_id]['file_name'].replace('.', '_gt.')))
        
        f.write('<tr>\n')
        f.write('<td><img src=\"{}\" alt="Modal" style="width:1200px;"></td>\n'.format(os.path.join('predictions', img_id_to_img[img_id]['file_name'].replace('.', '_modal.'))))
        f.write('<td><img src=\"{}\" alt="Amodal" style="width:1200px;"></td>\n'.format(os.path.join('predictions', img_id_to_img[img_id]['file_name'].replace('.', '_amodal.'))))
        f.write('<td><img src=\"{}\" alt="GT" style="width:1200px;"></td>\n'.format(os.path.join('predictions', img_id_to_img[img_id]['file_name'].replace('.', '_gt.'))))
        f.write('/<tr>\n')
    
    f.write(('</table>\n\n'
            '</body>\n'
            '</html>'))