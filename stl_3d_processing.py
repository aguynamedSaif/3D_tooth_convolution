from stltovoxel import slice
from stl import mesh

from PIL import Image
import tensorflow as tf

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import cv2


def array(input_file, resolution=50, resize_val=(128, 128)):
    meshes = []

    parallel = False
    colors = [(255, 255, 255)]

    mesh_obj = mesh.Mesh.from_file(input_file)
    org_mesh = np.hstack((mesh_obj.v0[:, np.newaxis], mesh_obj.v1[:, np.newaxis], mesh_obj.v2[:, np.newaxis]))
    meshes.append(org_mesh)

    scale, shift, shape = slice.calculate_scale_shift(meshes, resolution)
    voxels = np.zeros(shape[::-1], dtype=np.int8)

    for mesh_ind, org_mesh in enumerate(meshes):
        slice.scale_and_shift_mesh(org_mesh, scale, shift)
        vol = np.zeros(shape[::-1], dtype=bool)

        current_mesh_indices = set()
        z = 0
        for event_z, status, tri_ind in slice.generate_tri_events(org_mesh):
            while event_z - z >= 0:
                mesh_subset = [org_mesh[ind] for ind in current_mesh_indices]
                _, pixels = slice.paint_z_plane(mesh_subset, z, vol.shape[1:])
                vol[z] = pixels
                z += 1

            if status == 'start':
                assert tri_ind not in current_mesh_indices
                current_mesh_indices.add(tri_ind)
            elif status == 'end':
                assert tri_ind in current_mesh_indices
                current_mesh_indices.remove(tri_ind)

        voxels[vol] = mesh_ind + 1

    z_size = voxels.shape[0]

    size = str(len(str(z_size + 1)))

    # Black background
    colors = [(0, 0, 0)] + colors
    palette = [channel for color in colors for channel in color]

    data = []
    # Special case when white on black.
    for height in range(z_size):
        if colors == [(0, 0, 0), (255, 255, 255)]:
            img = Image.fromarray(voxels[height].astype('uint8'))
        else:
            img = Image.fromarray(voxels[height].astype('uint8'), mode='P')
            img.putpalette(palette)

        data.append(np.array(img.resize(resize_val)).astype('int32'))

    return np.array(data[:50])

def rotate(instance, angle):
    instance = ndimage.rotate(instance, angle, reshape=False)

    return instance


def plot_slices(num_rows, num_columns, width, height, data):
    data = np.reshape(data, (num_rows, num_columns, width, height))

    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]

    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)

    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )

    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j])
            axarr[i, j].axis("off")

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


def blur(data, kernel):
    """
    data -> multidimentional numpy array
    kernel -> (x,y) tuple consisting of odd numbers
    """

    gauss_blurr = []

    x,_,_ = data.shape
    for i in range(x):
        blurred = cv2.GaussianBlur(data[i].astype('float32'), kernel,cv2.BORDER_DEFAULT)
        gauss_blurr.append(blurred)

    return np.asarray(gauss_blurr)
