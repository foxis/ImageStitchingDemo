# ImageStitchingDemo
Created as an interview assignment.

# Usage
Number of arguments must be even and not less than 2.
usage:
stitch.py [test] <first_image_path> <second_image_path> ...
	The resulting image will be saved to <first_image_path>_<second_image_path>.png image into current directory
        If the first argument is test, then the images will be displayed.
	It is possible to provide multiple pairs of images which will be stitched in parallel(depends on the cpu core count)

# Method of operation
## Assumptions: 
 * both images are RGB
 * one image is larger than the other - usually RGB is smaller than IR
 * imageA will be the largest one
 * imageB will be the smaller one
 * FOV of both cameras are relatively similar. Otherwise some data might be missing due to larger misalignment
 * Image with higher resolution will be alligned relative to the smaller one

## Algorithm:

1. Find keypoints using SIFT/ORB features
2. Find keypoint matches 
3. Calculate homography matrix from that
4. Do perspective warp on imageA
5. Convert it to RGBA
6. Add alpha channel using imageB
7. Save result image as PNG

If test is provided as first argument - display intermediate images on screen

It is possible to select SIFT v.s. ORB features by setting use_sift to True
