import cv2
import os

def images_to_video(image_folder, output_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Common MP4 codec
    video = cv2.VideoWriter(output_name, fourcc, 30, (width, height)) 

    for image in sorted(images):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # cv2.destroyAllWindows()
    video.release()

# Example usage
image_folder = '/home/chengyeh/TAO-Amodal-Root/Amodal-Expander/demo/input_video'
output_name = 'input_video.mp4'  
images_to_video(image_folder, output_name)