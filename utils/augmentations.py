import cv2
import numpy as np
import sys
from scipy.interpolate import griddata


class RandomColorRotation(object):
    def __init__(self, **kwargs):
        # type: (Dict[...]) -> None
        self.kwargs = kwargs

    def __apply_random_color_rotation(self, img):
        # type: (numpy.ndarray) -> numpy.ndarray
        random_state = np.random.RandomState(self.kwargs.get("random_seed", None))
        shift = random_state.randint(0, 255)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = hsv[..., 0] + shift
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return img

    def __call__(self, img):
        # type: (numpy.ndarray) -> numpy.ndarray
        return self.__apply_random_color_rotation(img)


class TensmeyerBrightness(object):
    def __init__(self, foreground=0, background=0, sigma=30, **kwargs):
        # type: (...) -> None
        self.foreground = foreground
        self.background = background
        self.sigma = sigma
        self.kwargs = kwargs

    def __tensmeyer_brightness(self, img):
        # type: (numpy.ndarray) -> numpy.ndarray
        ret, th = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        th = (th.astype(np.float32) / 255)[..., None]

        img = img.astype(np.float32)
        img = img + (1.0 - th) * self.foreground
        img = img + th * self.background

        img[img > 255] = 255
        img[img < 0] = 0

        return img.astype(np.uint8)

    def __apply_tensmeyer_brightness(self, img):
        # type: (numpy.ndarray) -> numpy.ndarray
        random_state = np.random.RandomState(self.kwargs.get("random_seed", None))
        self.foreground = random_state.normal(0, self.sigma)
        self.background = random_state.normal(0, self.sigma)

        img = self.__tensmeyer_brightness(img)

        return img

    def __call__(self, img):
        # type: (numpy.ndarray) -> numpy.ndarray
        return self.__apply_tensmeyer_brightness(img)


class GridDistortion(object):
    def __init__(self, random_state=None, **kwargs):
        # type: (..., Dict[...]) -> None
        self.INTERPOLATION = {
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC
        }
        self.random_state = random_state
        self.kwargs = kwargs

    def __warp_image(self, img):
        if self.random_state is None:
            self.random_state = np.random.RandomState()

        w_mesh_interval = self.kwargs.get('w_mesh_interval', 25)
        w_mesh_std = self.kwargs.get('w_mesh_std', 3.0)

        h_mesh_interval = self.kwargs.get('h_mesh_interval', 25)
        h_mesh_std = self.kwargs.get('h_mesh_std', 3.0)

        interpolation_method = self.kwargs.get('interpolation', 'linear')

        h, w = img.shape[:2]

        if self.kwargs.get("fit_interval_to_image", True):
            # Change interval so it fits the image size
            w_ratio = w / float(w_mesh_interval)
            h_ratio = h / float(h_mesh_interval)

            w_ratio = max(1, round(w_ratio))
            h_ratio = max(1, round(h_ratio))

            w_mesh_interval = w / w_ratio
            h_mesh_interval = h / h_ratio
            ############################################

        # Get control points
        source = np.mgrid[0:h + h_mesh_interval:h_mesh_interval, 0:w + w_mesh_interval:w_mesh_interval]
        source = source.transpose(1, 2, 0).reshape(-1, 2)

        if self.kwargs.get("draw_grid_lines", False):
            if len(img.shape) == 2:
                color = 0
            else:
                color = np.array([0, 0, 255])
            for s in source:
                img[int(s[0]):int(s[0]) + 1, :] = color
                img[:, int(s[1]):int(s[1]) + 1] = color

        # Perturb source control points
        destination = source.copy()
        source_shape = source.shape[:1]
        destination[:, 0] = destination[:, 0] + self.random_state.normal(0.0, h_mesh_std,
                                                                         size=source_shape)
        destination[:, 1] = destination[:, 1] + self.random_state.normal(0.0, w_mesh_std,
                                                                         size=source_shape)

        # Warp image
        grid_x, grid_y = np.mgrid[0:h, 0:w]
        grid_z = griddata(destination, source, (grid_x, grid_y),
                          method=interpolation_method).astype(np.float32)
        map_x = grid_z[:, :, 1]
        map_y = grid_z[:, :, 0]
        warped = cv2.remap(img, map_x, map_y, self.INTERPOLATION[interpolation_method],
                           borderValue=(255, 255, 255))

        return warped

    def __call__(self, img, random_state=None, **kwargs):
        return self.__warp_image(img)


# NOT USED
class RandomBrightness(object):
    def __init__(self, brightness=0, contrast=1, b_range=(-50, 51), **kwargs):
        # type: (int, int, Tuple[int, int], Dict[...]) -> None
        self.brightness = brightness
        self.contrast = contrast
        self.b_range = b_range
        self.kwargs = kwargs

    def __increase_brightness(self, img):
        # type: (numpy.ndarray) -> numpy.ndarray
        img = img.astype(np.float32)
        img = img * self.contrast + self.brightness
        img[img > 255] = 255
        img[img < 0] = 0

        return img.astype(np.uint8)

    def __apply_random_brightness(self, img):
        # type: (numpy.ndarray) -> numpy.ndarray
        random_state = np.random.RandomState(self.kwargs.get("random_seed", None))
        self.brightness = random_state.randint(self.b_range[0], self.b_range[1])

        img = self.__increase_brightness(img)

        return img

    def __class_(self, img):
        # type: (numpy.ndarray) -> numpy.ndarray
        return self.__apply_random_brightness(img)


# to check
if __name__ == "__main__":
    input_image = sys.argv[1]
    output_image = sys.argv[2]
    img = cv2.imread(input_image)
    warp_image = GridDistortion()
    img = warp_image(img, draw_grid_lines=True)
    cv2.imwrite(output_image, img)
