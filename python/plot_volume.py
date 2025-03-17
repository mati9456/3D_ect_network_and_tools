import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors
import seaborn_image as isns

# plot eps_map
def plot_volume(volume, transparency_point=1.5, ax=None, bar=True, reduce=2, transparency=True):
    if len(volume.shape) == 3:

        volume = volume[::reduce, ::reduce, ::reduce]

        # get objects that are not transparent
        voxelarray = volume > transparency_point

        norm = matplotlib.colors.Normalize(vmin=transparency_point, vmax=3)
        colors = plt.cm.plasma(norm(volume))

        if transparency:
            transparency_scaling = ((volume-transparency_point)/(volume.max()-transparency_point))
            transparency_scaling = np.add(transparency_scaling, 0.4)
            transparency_scaling = np.clip(transparency_scaling, 0, 1)
            colors[:, :, :, 3] = np.multiply(colors[:, :, :, 3], transparency_scaling)
        
        # and plot everything
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(voxelarray, facecolors=colors, edgecolor=None)

        if bar:
            # colorbar
            m = cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
            m.set_array([])
            plt.colorbar(m, ax=ax)


    elif len(volume.shape) == 2:
        isns.imgplot(volume, ax=ax)
    else:
        print("input a 2d or 3d array")