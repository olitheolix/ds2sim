# -*- coding: utf-8 -*-

import os
import glob
import uuid
import pickle
import random
import tarfile
import tempfile
import Augmentor
import numpy as np
import urllib.request

from PIL import Image


class RotateScaleShift(Augmentor.Operations.Operation):
    def __init__(self, probability, min_scale=0.2, max_scale=1):
        super().__init__(probability)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def perform_operation(self, img):
        # Create NumPy array with correct dimensions (will hold the output image).
        out = 0 * np.array(img, np.uint8)
        h, w, _ = out.shape
        assert w == h

        phi = 360 * np.random.random()
        img = img.rotate(phi, expand=False, resample=Image.BICUBIC)

        # Scale the image (new image is always smaller or equal than original).
        scale = self.min_scale + (self.max_scale - self.min_scale) * np.random.random()
        ws = int(scale * w)
        img = img.resize((ws, ws))
        del scale

        # Randomly pick a ws x ws region in the output image.
        x0 = np.random.randint(0, w - ws + 1)
        y0 = np.random.randint(0, w - ws + 1)
        x1, y1 = x0 + ws, y0 + ws

        # Place the shrunk image in the full sized output image, and return it.
        out[y0:y1, x0:x1, :] = np.array(img, np.uint8)
        return Image.fromarray(out)


def unpackCIFAR10(dst_path):
    src_path = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(src_path, '..', 'dataset')
    fname_cifar10 = os.path.join(src_path, 'cifar10.tar.gz')

    # Download the CIFAR10 dataset, unless it already exists.
    if not os.path.exists(fname_cifar10):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        print(f'Downloading CIFAR10 from <{url}>')
        urllib.request.urlretrieve(url, filename=fname_cifar10)
        f = urllib.request.urlopen(url)
        open(fname_cifar10, 'wb').write(f.read())
        del url, f

    # Open TAR file and find the pickled feature files.
    tar = tarfile.open(fname_cifar10, 'r:gz')
    members = [_ for _ in tar.getmembers()
               if _.name.startswith('cifar-10-batches-py/data_batch_')]

    # Load all features in CIFAR10. We do not care about the labels, since
    # CIFAR10 will only be used as the 'background' label.
    dims = (3, 32, 32)
    all_labels = []
    all_features = np.zeros((0, np.prod(dims)), np.uint8)
    for member in members:
        # Unpickle the dictionary and unpack the features and labels.
        data = pickle.load(tar.extractfile(member), encoding='bytes')
        labels, features = data[b'labels'], data[b'data']

        # Sanity check.
        assert len(labels) == features.shape[0]
        assert features.shape[1] == np.prod(dims)

        # Add the latest features to our stash.
        all_features = np.vstack([all_features, features])
        all_labels.extend(labels)

    # Limit the set to 1000 random images.
    all_labels = np.array(all_labels, np.int32)
    p = np.random.permutation(len(all_labels))
    all_features = all_features[p, :]
    all_labels = all_labels[p]
    all_features = all_features[:1000]
    all_labels = all_labels[:1000]

    dst_path = os.path.join(dst_path, 'cifar10')

    # Do nothing if the folder already exists.
    try:
        os.mkdir(dst_path)
    except FileExistsError:
        pass

    # Rescale all images.
    for idx, img in enumerate(all_features):
        img = np.reshape(img, dims)
        img = np.rollaxis(img, 0, 3)
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = img.resize((128, 128), resample=Image.BICUBIC)

        img.save(os.path.join(dst_path, f'{idx:04d}.jpg'))


def unpackDS2(dst_path):
    src_path = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(src_path, '..', 'dataset')
    fname_ds2 = os.path.join(src_path, 'ds2.tar.gz')

    # Download the DS@ dataset, unless it already exists.
    if not os.path.exists(fname_ds2):
        url = 'https://s3-ap-southeast-2.amazonaws.com/olitheolix/dataset/ds2.tar.gz'
        print(f'Downloading DS2 Dataset from <{url}>')
        urllib.request.urlretrieve(url, filename=fname_ds2)
        f = urllib.request.urlopen(url)
        open(fname_ds2, 'wb').write(f.read())
        del url, f

    tar = tarfile.open(fname_ds2, 'r:gz')
    tar.extractall(dst_path)
    tar.close()


def augmentDS2(src_path, dst_path, N):
    for i in range(10):
        src = os.path.join(src_path, str(i))
        dst = os.path.join(dst_path, str(i))

        p = Augmentor.Pipeline(src, dst, 'JPEG')
        p.add_operation(RotateScaleShift(probability=1))
        p.flip_left_right(probability=0.5)
        p.flip_top_bottom(probability=0.5)
        p.sample(N)


def augmentCIFAR10(src_path, dst_path, N):
    src = os.path.join(src_path, 'cifar10')
    dst = os.path.join(dst_path, 'cifar10')

    p = Augmentor.Pipeline(src, dst, 'JPEG')
    p.random_distortion(1, 5, 5, 5)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.sample(N)


def createSceneImage(data_path):
    N = 10
    dims = (128, 128, 3)

    tilesize = 150
    width, height = N * tilesize, 10 * tilesize
    out_img = np.zeros((width, height, dims[2]), np.uint8)
    del width, height

    # Each row features cubes with the same number.
    for row in range(10):
        # Get N random pictures of cubes with number 'row' on it.
        fnames = glob.glob(os.path.join(data_path, str(row), '*'))
        fnames = random.sample(fnames, N)

        # Arrange the N cubes in a line.
        for col, fname in enumerate(fnames):
            # Load the 128x128x3 image and convert it to a NumPy array.
            img = Image.open(fname)
            img = np.array(img, np.uint8)
            assert img.shape == dims

            # Determine where to put the cube in the final image.
            x0 = col * tilesize + (tilesize - dims[1]) // 2
            y0 = row * tilesize + (tilesize - dims[0]) // 2
            x1, y1 = x0 + dims[1], y0 + dims[0]

            # Insert the cube.
            out_img[y0:y1, x0:x1, :] = img

    # Ensure the destination folder exists.
    dst_path = os.path.join(data_path, 'scene')
    try:
        os.mkdir(dst_path)
    except FileExistsError:
        pass

    # Save the image as JPEG with a random name.
    out_img = Image.fromarray(out_img)
    fname = os.path.join(dst_path, f'{uuid.uuid1()}.jpg')
    out_img.save(fname)


def main():
    num_samples = 100
    data_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(data_path, '..', 'dataset')

    with tempfile.TemporaryDirectory() as tdir:
        unpackDS2(tdir)
        unpackCIFAR10(tdir)

        augmentDS2(tdir, data_path, num_samples)
        augmentCIFAR10(tdir, data_path, num_samples)

    for i in range(num_samples):
        createSceneImage(data_path)


if __name__ == '__main__':
    main()
