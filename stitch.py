#!/usr/bin/python
from sys import argv
from multiprocessing import Process, Pool, cpu_count
import numpy as np
import imutils
import cv2
from os.path import basename


class StitchingException(Exception):
    pass


class StitchingTask(object):
    """
        Takes two images(one RGB and one BW), stitches them so that
        original RGB image remains, and the second one goes into A channel of
        the result image. Saves the image.
    """

    def __init__(self, images, result):
        self.image1, self.image2 = images
        self.result = result

    def run(self):
        try:
            imageA = cv2.imread(self.image1)
            imageB = cv2.imread(self.image2)

            if hasattr(self, 'drawInputs'):
                self.drawInputs(imageA, imageB)

            if imageA.shape[0] < imageB.shape[0]:
                imageA, imageB = imageB, imageA

            scale_width = imageA.shape[1] / imageB.shape[1]
            scale_height = imageA.shape[0] / imageB.shape[0]
            scale = min(scale_width, scale_height) # percent of original size
            width = int(imageB.shape[1] * scale)
            height = int(imageB.shape[0] * scale)
            dim = (width, height)
            imageB = cv2.resize(imageB, dim, interpolation=cv2.INTER_AREA)

            result = self.stitch(imageA, imageB)
            cv2.imwrite(self.result, result)
        except StitchingException as e:
            print "Stitching was not possible due to: ", str(e)
        except Exception as e:
            print "Some unhandled error occured"
            raise  # for stack trace

    """
    The code below was taken from https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
    and tailored slightly to my needs
    """
    def stitch(self, imageA, imageB, ratio=0.75, reprojThresh=4.0):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        kpsA, featuresA = self.detectAndDescribe(imageA)
        kpsB, featuresB = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            raise StitchingException("No matches were found between images.")

        matches, H, status = M

        #TODO calculate resulting image size
        # create RGBA image
        # split RGB image
        # merge

        # otherwise, apply a perspective warp to stitch the images
        # together
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1], imageA.shape[0]))
        rgba = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
        bw = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
        rgba[0:imageB.shape[0], 0:imageB.shape[1], 3] = bw

        if hasattr(self, 'drawMatches'):
            self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status, rgba)

        # return the stitched image
        return rgba

    def detectAndDescribe(self, image):
        # detect and extract features from the image
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return kps, features

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        raise StitchingException("No homography could be computed")


class StitchingTestTask(StitchingTask):
    """
    Simple test to see if images are loaded and in what order by displaying them
    """
    def stitch(self, imageA, imageB, ratio=0.75, reprojThresh=4.0):
        print "First image shape:", imageA.shape
        print "Second image shape:", imageB.shape
        print "saved image name:", self.result
        super(StitchingTestTask, self).stitch(imageA, imageB, ratio, reprojThresh)
        cv2.waitKey(0)

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status, result):
        print "Result image shape:", result.shape

        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        cv2.imshow("Keypoint Matches", vis)
        cv2.imshow("Result", result)

    def drawInputs(self, imageA, imageB):
        cv2.imshow("Image A", imageA)
        cv2.imshow("Image B", imageB)


def caller_helper(worker):
    """ Required by multiprocessing to execute pools. Lambdas don't work sadly. """
    return worker.run()


def main(args, test=False):
    """
    main function to execute stitching.
    Will parse args, generate result image filename, create and execute the process pool.

    :args: a list of source image paths
    :test: indicates whether it is a test to see if multiprocessing works as expected and images are loaded
    """
    pool = Pool(cpu_count())

    pairs = [args[i:i + 2] for i in range(0, len(args), 2)]
    results = ["{0}_{1}.png".format(basename(pair[0]), basename(pair[1])) for pair in pairs]

    klass = StitchingTestTask if test else StitchingTask
    pool.map(caller_helper, [klass(*args) for args in zip(pairs, results)])


if __name__ == "__main__":
    fn = 2 if len(argv) > 1 and argv[1] == 'test' else 1

    if len(argv[fn:]) < 2 or len(argv[fn:]) % 2:
        print "Number of arguments must be even and not less than 2."
        print "usage:"
        print "stitch.py [test] <first_image_path> <second_image_path> ..."
        print "The resulting image will be saved to <first_image_path>_<second_image_path>.png image into current directory"
        print "if the first argument is test, then the images will be displayed."
        exit(1)

    main(argv[fn:], test=fn == 2)