import argparse
import phys as phys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

IMG_WIDTH = 512

normalization_modes = {
    'none': colors.NoNorm(),
    'log': colors.LogNorm(vmin=0.001, vmax=1)
}


def main():
    parser = argparse.ArgumentParser(description='Visualize airy disk.')
    parser.add_argument('lambda', type=float, help='Wavelength')
    parser.add_argument('a', type=float, help='Aperture wavelength')
    parser.add_argument('theta', type=float,
                        help='Theta from center of displayed image to edge')
    parser.add_argument('i0', type=float, default=1,
                        help='Intensity at center')
    parser.add_argument('norm', type=str, default='none',
                        help='Normalization mode')

    args = parser.parse_args()
    print('args: {}'.format(args))

    view_theta = args.theta
    wavelength = getattr(args, 'lambda')
    a = args.a
    i0 = args.i0
    norm = normalization_modes[args.norm]

    # Create np array
    arr = np.ndarray((IMG_WIDTH, IMG_WIDTH))

    # populate array
    arr = phys.airy_arr(wavelength, a, i0=i0, arr=arr, view_theta=view_theta)

    plt.imshow(arr, cmap='gray', norm=norm)
    plt.show()


if __name__ == '__main__':
    main()
