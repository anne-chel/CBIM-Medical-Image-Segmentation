import argparse
import yaml
import cv2
import torch
import imutils
import numpy as np
import SimpleITK as sitk
from imutils import contours
import matplotlib.pyplot as plt

def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def is_between(a, c, b):
    return distance(a, c) + distance(c, b) == distance(a, b)

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Function that takes as input the ED and ES volumes of the ventricle and computes the EF
def compute_EF(ED_volume, ES_volume):
    return (ED_volume - ES_volume) / ED_volume * 100

# Method tha takes as input the dimensions (width and length) of the ventricle
# on systole and siastole for each chamber view and computes the volume of the ventricle
def compute_volume(ventricle_widths, ventricle_length):
    """
    Takes as input the dimensions (width and length) of the ventricle
    on systole and siastole for each chamber view and computes the volume of the ventricle
    ---------
    input
    ventricle_widths: list of lists of floats
        List of lists of floats containing the width of the ventricle at each point in the ventricle
        for each chamber view
    ventricle_length: float
        Length of the ventricle
    ---------
    output
    ventricle_volume: float
        Volume of the ventricle
    """
    # Make the lists the same length
    # Find the minimum length of the diastole and systole lists
    max_length = max([len(ventricle_widths[x]) for x in range(len(ventricle_widths))])
    assert len(ventricle_widths) == 2 and max_length > 0
    # Make the lists the same length
    for i in range(2):
        if len(ventricle_widths[i]) < max_length:
            # Copy the first value of the list in the beginning of the list
            # until the list is the same length as the other lists
            while len(ventricle_widths[i]) < max_length:
                ventricle_widths[i].insert(0, ventricle_widths[i][0])
    width_array = np.array(ventricle_widths[:2])
    volume = np.sum(np.pi * width_array[0] * width_array[1] * ventricle_length)
    return volume

def get_widths(mask_2chamber, mask_4chamber, disk_size=25, mode="ED"):
    """
    Function that takes as input the segmentation mask of the 2chamber and 4chmber view
    splits the vetricle into small disks and calculates the width of the ventricle at
    each slice. It returns a list of the ventricle widths for each mask and the length of the ventricle
    ----------------
    input:
    mask_2chamber: segmentation mask of the 2chamber view
    mask_4chamber: segmentation mask of the 4chamber view
    disk_size: size of the disk used to split the ventricle into small disks
    returns:
    ventricle_widths: list of two lists containing the ventricle widths for each mask size
    ventricle_length: length of the ventricle
    """

    ventricle_widths = []
    for k, image in enumerate([mask_2chamber, mask_4chamber]):
        ventricle_segment = np.array((image == 1), dtype=np.uint8)

        # Normalise image
        normed_img = (ventricle_segment - ventricle_segment.min()) / (
            ventricle_segment.max() - ventricle_segment.min()
        )
        normed_img = (ventricle_segment * 255).astype(np.uint8)
        # Make a figure with 2 subplots showing image and normalised image

        # Find the edges in the image using canny detector
        edged = cv2.Canny(normed_img, 50, 200)

        # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)

        # sort the contours from left-to-right and initialize the
        # 'pixels per metric' calibration variable
        (cnts, _) = contours.sort_contours(cnts)

        # From the contours, find the contour with the largest area
        # and keep only the contour with the largest area
        cnts = [max(cnts, key=cv2.contourArea)]

        # loop over the contours individually
        for c in cnts:
            print("JOEHOEEEEEE")
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 2:
                continue
            # compute the rotated bounding box of the contour
            orig = image.copy()
            hull = cv2.convexHull(c).squeeze(1)

            # Find the highest point (lowest y value) in the hull
            highest_idx = np.argmin(hull[:, 1])
            highest_point = hull[highest_idx]

            lines = cv2.HoughLinesP(
                edged,
                rho=1,
                theta=1 * np.pi / 180,
                threshold=50,
                minLineLength=15,
                maxLineGap=80,
            )
            angles = []
            lines_of_interest = []
            if lines is not None:
                N = lines.shape[0]
                for i in range(N):
                    x1 = lines[i][0][0]
                    y1 = lines[i][0][1]
                    x2 = lines[i][0][2]
                    y2 = lines[i][0][3]
                    middle_x = int(np.floor(x1 + 0.5 * (x2 - x1)))
                    middle_y = int(np.floor(y1 + 0.5 * (y2 - y1)))
                    kernel = np.array([[-2, -1, 0, 1, 2]])
                    window_x = middle_x + kernel
                    window_v = middle_y + kernel
                    values = set()
                    for x in range(-10, 11):
                        for y in range(-10, 11):
                            # Check that the window is within the image
                            if (
                                (middle_y + y) < 0 or (middle_y + y) >= image.shape[0]
                            ) or (
                                (middle_x + x) < 0 or (middle_x + x) >= image.shape[1]
                            ):
                                continue
                            else:
                                values.add(image[middle_y + y, middle_x + x])
                    if values.issubset([1, 3]):
                        # Draw the line on the image
                        cv2.line(
                            orig,
                            [int(highest_point[0]), int(highest_point[1])],
                            [middle_x, middle_y],
                            (255, 0, 0),
                            1,
                        )
                        # Append the line to the list of lines of interest
                        lines_of_interest.append([(x1, y1), (x2, y2)])
                        # Save the point middle point
                        mid_bottom = (middle_x, middle_y)

            # Separate the hull into smaller regions to compute th width of the ventricle
            min_y = np.min(hull[:, 1])
            max_y = np.max(hull[:, 1])
            # Start from the bottom of the hull and move up by 10 pixels
            # until the top of the hull is reached
            # For each region, compute the width of the ventricle
            # and add it to the list
            y_bottom = max_y
            # Calculate the length of the ventricle
            if len(lines_of_interest) == 0:
                # Get the bottom right point of the contour
                # Find the point with the highest x value and highest y value
                bottom_right_idx = np.argmax(hull[:, 0] + hull[:, 1])
                bottom_right = hull[bottom_right_idx]
                # Get the bottom left point of the contour
                # Find the point with the lowest x value and highest y value
                bottom_left_idx = np.argmin(hull[:, 0] - hull[:, 1])
                bottom_left = hull[bottom_left_idx]
                # Find which point of the contour is closest to the middle of the bottom left and right points
                mid_bottom = np.array((bottom_left + bottom_right) / 2)

            if k == 1:
                ventricle_length = np.sqrt(
                    (highest_point[0] - mid_bottom[0]) ** 2
                    + (highest_point[1] - mid_bottom[1]) ** 2
                )

            current_ventricle_widths = []
            conts = c.squeeze(1)

            for y in range(y_bottom - disk_size, min_y, -disk_size):
                # Find the points in the hull that are at the current y value
                # and find the middle point to compute the width
                hull_y = conts[conts[:, 1] <= y_bottom]
                hull_y = hull_y[hull_y[:, 1] >= y]
                # Check if there are any points in the hull at the current y value
                if len(hull_y) > 0:
                    # Find the pointthat is closest to the middle between the top and bottom
                    # of the current region (y and y_bottom) and compute the width from the
                    # left and right side of the ventricle
                    middle_y = int(np.floor(y + 0.5 * (y_bottom - y)))
                    middle_x = int(np.floor(np.mean(hull_y[:, 0])))

                    # Find which two points (left and right) of the hull are closest ventricle_widthsto
                    # the middle point and compute the width from the
                    # left and right side of the ventricle
                    left_point = hull_y[
                        np.argmin(
                            np.cross(hull_y - middle_x, hull_y - middle_y)
                            / np.linalg.norm(hull_y - middle_x)
                        )
                    ]
                    right_point = hull_y[
                        np.argmax(
                            np.cross(hull_y - middle_x, hull_y - middle_y)
                            / np.linalg.norm(hull_y - middle_x)
                        )
                    ]
                    # Width of the ventricle is the distance between the left and right point
                    width = np.sqrt(
                        (left_point[0] - right_point[0]) ** 2
                        + (left_point[1] - right_point[1]) ** 2
                    )
                    # Add the width to the list
                    current_ventricle_widths.append(width)

                    # Update the left and right points
                    current_left = left_point
                    current_right = right_point
                else:
                    # Print hull_y and the current y value
                    print(f"hull_y: {hull_y}, y: {y}, y_bottom: {y_bottom}")
                    # If there are no points in the hull at the current y value
                    # then the width of the ventricle is the same as the previous
                    # ventricle_widths[k].append(ventricle_widths[k][-1])
                    # Move the bottom of the region up by 10 pixels
                    current_left[1] -= disk_size
                    current_right[1] -= disk_size

                # Update the bottom y valuebottom of the regionbottom ofbottom of the region the region
                y_bottom = y

        # Add the list of ventricle widths to the list of ventricle widths
        ventricle_widths.append(current_ventricle_widths)

    return ventricle_widths, ventricle_length

# Method tha takes as input the dimensions (width and length) of the ventricle
# on systole and siastole for each chamber view and computes the volume of the ventricle
def compute_volume(ventricle_widths, ventricle_length):
    """
    Takes as input the dimensions (width and length) of the ventricle
    on systole and siastole for each chamber view and computes the volume of the ventricle
    ---------
    input
    ventricle_widths: list of lists of floats
        List of lists of floats containing the width of the ventricle at each point in the ventricle
        for each chamber view
    ventricle_length: float
        Length of the ventricle
    ---------
    output
    ventricle_volume: float
        Volume of the ventricle
    """
    # Make the lists the same length
    # Find the minimum length of the diastole and systole lists
    max_length = max([len(ventricle_widths[x]) for x in range(len(ventricle_widths))])
    assert len(ventricle_widths) == 2 and max_length > 0
    # Make the lists the same length
    for i in range(2):
        if len(ventricle_widths[i]) < max_length:
            # Copy the first value of the list in the beginning of the list
            # until the list is the same length as the other lists
            while len(ventricle_widths[i]) < max_length:
                ventricle_widths[i].insert(0, ventricle_widths[i][0])
    width_array = np.array(ventricle_widths[:2])
    volume = np.sum(np.pi * width_array[0] * width_array[1] * ventricle_length)
    return volume

# Create a function that takes as input the image mask of the 2chamber and 4chamber views
# 1. Calculates the EDV and ESV from the 2chamber and 4chambre view
# 2. Calculates the EF from the EDV and ESV
def calculate_volume(images=None, true_ef=None):

    #in order: 2CH_ED, 2CH_ES, 4CH_ED, 4CH_ES
    mask_2chamber_ED = images[0].squeeze().detach().numpy()
    mask_4chamber_ED = images[2].squeeze().detach().numpy()

    mask_2chamber_ES = images[1].squeeze().detach().numpy()
    mask_4chamber_ES = images[3].squeeze().detach().numpy()

    ventricle_ED_widths, ventricle_length = get_widths(
            mask_2chamber_ED, mask_4chamber_ED, disk_size=5, mode="ED")

    ventricle_ES_widths, _ = get_widths(
                mask_2chamber_ES, mask_4chamber_ES, disk_size=5, mode="ES")
        
    # Compute the ED and ES volumes
    volume_diastole = compute_volume(ventricle_ED_widths, ventricle_length)
    volume_systole = compute_volume(ventricle_ES_widths, ventricle_length)
    # Calculate the ejection fraction
    ejection_fraction = compute_EF(volume_diastole, volume_systole)

    if true_ef != None:
        error = abs(true_ef - ejection_fraction)
        return error, ejaction_fraction
    else:
        return ejection_fraction