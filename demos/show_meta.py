#!env python
# -*- coding: utf-8 -*-
"""
Draw a bounding box around the visible cubes in every frame.

This script shows how to parse the pickled meta data and extract the BBoxes and
their labels. It takes the image paths as an argument and will *overwrite* all
images with new ones that contain the BBoxes.
"""
import bz2
import json
import argparse

import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parseCmdline():
    """Parse the command line arguments."""
    # Create a parser and program description.
    parser = argparse.ArgumentParser(
        description='Show the specified image and draw the BBoxes and labels.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('jpg', type=str, help='Path to JPG file')
    return parser.parse_args()


def main():
    param = parseCmdline()

    # Load the meta data with the BBox information.
    fname_meta = param.jpg[:-4] + '-meta.json.bz2'
    meta = json.loads(bz2.open(fname_meta, 'rb').read().decode('utf8'))

    # Undo JSON's int->str conversion for dict keys.
    bb_data = {int(k): v for k, v in meta['bb_data'].items()}
    int2name = {int(k): v for k, v in meta['int2name'].items()}
    obj_pixels = {int(k): v for k, v in meta['obj-pixels'].items()}
    objID2label = {int(k): v for k, v in meta['objID2label'].items()}

    # For convenience.
    objID_at_pixel = np.array(meta['objID-at-pixel'], np.int32)

    # Number of classes in entire data set (does not mean all of them are
    # present in the image).
    num_classes = len(int2name)

    # Matplotlib constants to draw pretty boxes and text labels.
    rect_opts = dict(linewidth=1, facecolor='none', edgecolor='g')
    txt_opts = dict(
        bbox={'facecolor': 'black', 'pad': 0},
        fontdict=dict(color='white', size=12, weight='normal'),
        horizontalalignment='center', verticalalignment='center'
    )

    # Display the JPG file.
    plt.figure()
    ax = plt.subplot(1, 3, 1)
    plt.imshow(Image.open(param.jpg))

    # Add all BBoxes.
    for objID, bb_data in bb_data.items():
        # Fact: the object is in the camera frustrum. However, it may be
        # (partially) obscured by another object. To find out we compute the
        # ratio of pixel the object *would* occupy if it were the only object
        # in the scene. Then we determine how many of the visible pixels
        # still belong to that object.
        num_pixels_total = len(np.nonzero(obj_pixels[objID])[0])
        num_visible = np.count_nonzero(objID_at_pixel == objID)
        visible_rat = num_visible / num_pixels_total

        # Draw a BBox if the object is at least 25% visible.
        if visible_rat > 0.25:
            # Unpack the BBox, compute width and height, and draw it.
            x0, y0, x1, y1 = bb_data['bbox']
            w = x1 - x0
            h = y1 - y0
            ax.add_patch(patches.Rectangle((x0, y0), w, h, **rect_opts))

            # Add a test label if the object is at least 50% visible.
            if visible_rat > 0.5:
                name = int2name[bb_data['label']]
                ax.text(x0 + w / 2, y0, f' {name} ', **txt_opts)

    # Show object ID for each pixel. Zero means no objec occupies that pixel
    # (ie it is background).
    plt.subplot(1, 3, 2)
    plt.imshow(objID_at_pixel, clim=[0, meta['num_cubes']])
    plt.title('Object ID')

    # For each non-zero pixel, map the object ID to its label.
    p_labels = np.zeros_like(objID_at_pixel)
    for idx in zip(*np.nonzero(objID_at_pixel)):
        p_labels[idx] = objID2label[objID_at_pixel[idx]]

    plt.subplot(1, 3, 3)
    plt.imshow(p_labels, clim=[0, num_classes])
    plt.title('Object Label')

    plt.show()


if __name__ == '__main__':
    main()
