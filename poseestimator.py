import cv2
import cv2.cv as cv
import numpy as np

ESC = 27
ENTER = 13
title_live = "Live feed"
title_im1 = "Image 1"
title_im2 = "Image 2"

def main_loop():

    with webcam() as w:

        key = 22
        while key != ESC:
            key = cv2.waitKey(10)
            show_video_feed(w.video)

            if key == ENTER:
                im1, im2 = pipeline(w.read())
                show_two_images(im1, im2)

class webcam:
    def __enter__(self):
        self.video = cv2.VideoCapture(0)
        if self.video.isOpened():
            result, self.permaframe = self.video.read()
        else:
            raise Exception("Couldn't get first frame.")
        return self

    def __exit__(self, type, value, traceback):
        cv2.destroyAllWindows()
        self.video.release()

    def read(self):
        temp = self.permaframe
        self.permaframe = self.video.read()[1]
        return (self.permaframe, temp)


def show_video_feed(videocapture):
    cv2.namedWindow(title_live)
    cv2.imshow(title_live, videocapture.read()[1])

def show_two_images(im1, im2):
    cv2.namedWindow(title_im1)
    cv2.namedWindow(title_im2)
    cv2.imshow(title_im1, im1)
    cv2.imshow(title_im2, im2)

def convert_to_grayscale(im1, im2):
    imgr1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    imgr2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    return (imgr1, imgr2)


def get_features(im1, im2):

    surf = cv2.SURF()
    k1, d1 = surf.detectAndCompute(im1, None)
    k2, d2 = surf.detectAndCompute(im2, None)
    im1 = cv2.drawKeypoints(im1, k1)
    return (im1, k1, d1, im2, k2, d2)


def match(d1, d2):
    # Black magic. Need a much better understanding of OpenCV internals, plus some machine learning
    # http://stackoverflow.com/questions/12621535/how-to-compare-surf-features-in-python-opencv2-4
    # I'm not going to bother learning this because I expect they are going to completely redo this API soon because it is so terrible

    def match_flann(desc1, desc2, r_threshold = 0.6):
        FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
        flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)
        flann = cv2.flann_Index(desc2, flann_params)
        idx2, dist = flann.knnSearch(desc1, 2, params = {}) # bug: need to provide empty dict
        mask = dist[:,0] / dist[:,1] < r_threshold
        idx1 = np.arange(len(desc1))
        pairs = np.int32( zip(idx1, idx2[:,0]) )
        return pairs[mask]

    return match_flann(d1, d2, 0.8)


def homo(k1, k2, pairs):
    p0 = [target.keypoints[m.trainIdx].pt for m in matches]
    p1 = [self.frame_points[m.queryIdx].pt for m in matches]
    p0, p1 = np.float32((p0, p1))
    H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)


def pipeline((im1, im2)):
    im1, im2 = convert_to_grayscale(im1, im2)
    im1, k1, d1, im2, k2, d2 = get_features(im1, im2)

    pairs = match(d1, d2)

    import pdb;
#    pdb.set_trace()

    # SURF to find keypoints
    # Match keypoints
    # Find homography
    # Decompose homography
    # Smooth output
    return im1, im2




if __name__ == '__main__':
    main_loop()


