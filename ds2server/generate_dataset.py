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


def makeFolder(dir_name):
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass


def unpackCIFAR10(tmp_dir, cache_dir):
    makeFolder(cache_dir)
    fname_cifar10 = os.path.join(cache_dir, 'cifar10.tar.gz')

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

    tmp_dir = os.path.join(tmp_dir, 'cifar10')

    # Do nothing if the folder already exists.
    try:
        os.mkdir(tmp_dir)
    except FileExistsError:
        pass

    # Rescale all images.
    for idx, img in enumerate(all_features):
        img = np.reshape(img, dims)
        img = np.rollaxis(img, 0, 3)
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = img.resize((128, 128), resample=Image.BICUBIC)

        img.save(os.path.join(tmp_dir, f'{idx:04d}.jpg'))


def unpackDS2(tmp_dir, cache_dir):
    makeFolder(cache_dir)

    path = os.path.dirname(os.path.abspath(__file__))
    fname_tar = os.path.join(path, 'dataset', 'ds2.tar.gz')

    tar = tarfile.open(fname_tar, 'r:gz')
    tar.extractall(tmp_dir)
    tar.close()


def augmentDS2(src_path, dst_path, N):
    for i in range(10):
        src = os.path.join(src_path, f'{i:02d}')
        dst = os.path.join(dst_path, f'{i:02d}')

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
        fnames = glob.glob(os.path.join(data_path, f'{row:02d}', '*'))
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
    data_dir = os.path.join(os.getcwd(), 'dataset')
    makeFolder(data_dir)

    with tempfile.TemporaryDirectory() as tdir:
        unpackDS2(tdir, 'cache')
        unpackCIFAR10(tdir, 'cache')

        augmentDS2(tdir, data_dir, num_samples)
        augmentCIFAR10(tdir, data_dir, num_samples)

    for i in range(num_samples):
        createSceneImage(data_dir)


if __name__ == '__main__':
    main()
