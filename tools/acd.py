import cv2
import numpy as np
import glob

def ACD(filename):
    """
    Calculates Average Content Distance for the given short video
    
    Parameters:
    -----------
        filename : str or path-like - path to .mp4 or .gif file
    """
    #print(filename)
    ViCap = cv2.VideoCapture(filename)

    frames = []
    success = True
    while success:
        success, image = ViCap.read()
        if success: 
            frames += [image]

    ViCap.release()
    cv2.destroyAllWindows()
    frames = np.array(frames, dtype='int32')
    
    assert len(frames) > 0, \
    "Sth went wrong, no frames were extracted"
        
    N = np.multiply.reduce(frames.shape[1:-1])
    res = np.mean(
            np.linalg.norm(
              np.diff(
                np.einsum('ijkl->il', frames), 
              axis=0) / N, 
            axis=1)
          ) 

    return res

if __name__ == "__main__":
    path = "G:/training_datasets/acd/*.mp4"
    dataset = glob.glob(path)
    for v in dataset:
        res = ACD(v)
        print("ACD score:", res)