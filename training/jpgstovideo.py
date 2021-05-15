import cv2
import os
import re
import sys

fps = 32

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def convert(image_folder, output_video, format):
    images = [img for img in os.listdir(image_folder) if img.endswith(format)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    #video = cv2.VideoWriter(output_video, 0, 1, (width,height), fps=fps)
    video = cv2.VideoWriter(filename=output_video, 
        fourcc=cv2.VideoWriter_fourcc(*'DIVX'),         
        fps=fps,                                     
        frameSize=(width, height))
    images.sort(key=natural_keys)
    for image in images:
        print("image:", image)
        video.write(cv2.imread(os.path.join(image_folder, image)))
        #video.write(image)
    
    cv2.destroyAllWindows()
    video.release()

def rename(image_folder, output_video, format):
    images = [img for img in os.listdir(image_folder) if img.endswith(format)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    #images.sort(key=natural_keys)
    for image in images:
        new_name = image.replace("_01", "")
        print("new_name:", new_name)
        cv2.imwrite(output_video+"/%s" % new_name, cv2.imread(os.path.join(image_folder, image)))
    

def videoToFrames():
    video_filenames = os.listdir("trainvideos")
    length = len(video_filenames)
    for i in range(length):
        print("video_filenames: ", video_filenames[i])
        video = video_filenames[i]
        outputDir = "frames/" + video.rpartition('.')[0]
        print("outputDir: ", outputDir)
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        toFrames("trainvideos/" + video, outputDir)

def labelFrames():
    frame_filenames = os.listdir("frames")
    for frame in frame_filenames:
        print("Label video: ", frame)
        labelAll("frames/" + frame,'labeled/')

if __name__ == '__main__':
    path = "G:/training_datasets/ffhq_" + sys.argv[1] + ".mp4"
    print("save to ", path)
    convert("G:/training_datasets/interpolate_ffhq", path, ".png")
    print("Created a video!")
    
    