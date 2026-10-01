from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.transform import resize
import skimage.io as skio
from PIL import Image

root = Path(r'C:\Users\eshap\cs180-fa26\project2-filters')
out_root = Path(r'C:\Users\eshap\esha-website\images\proj2')
out_root.mkdir(parents=True, exist_ok=True)

D_x = np.array([[-1, 0, 1]])
D_y = np.array([[-1], [0], [1]])

def convolution_two_loops(image, kernel):
    im_height, im_width = image.shape
    k_height, k_width = kernel.shape
    output = np.zeros_like(image, dtype=float)
    kernel_flipped = np.flip(kernel)
    pad_height = k_height // 2
    pad_width = k_width // 2
    padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    for row in range(im_height):
        for col in range(im_width):
            patch = padded[row:row + k_height, col:col + k_width]
            output[row, col] = np.sum(patch * kernel_flipped)
    return output


def unsharp_mask(image, kernel_size=5, sigma=0, alpha=1.0):
    gaussian_1d = cv2.getGaussianKernel(kernel_size, sigma)
    G = gaussian_1d @ gaussian_1d.T
    identity = np.zeros_like(G)
    center = kernel_size // 2
    identity[center, center] = 1
    unsharp_kernel = (1 + alpha) * identity - alpha * G
    if image.ndim == 2:
        sharpened = convolve2d(image, unsharp_kernel, mode='same', boundary='fill')
    else:
        sharpened = np.zeros_like(image, dtype=float)
        for c in range(image.shape[2]):
            sharpened[:, :, c] = convolve2d(image[:, :, c], unsharp_kernel, mode='same', boundary='fill')
    return sharpened


def hybrid_image(low_image, high_image, kernel_size=21, sigma=5, high_weight=1.5):
    if low_image.ndim == 3:
        low_image = 0.2989 * low_image[:, :, 0] + 0.5870 * low_image[:, :, 1] + 0.1140 * low_image[:, :, 2]
    if high_image.ndim == 3:
        high_image = 0.2989 * high_image[:, :, 0] + 0.5870 * high_image[:, :, 1] + 0.1140 * high_image[:, :, 2]

    gaussian_1d = cv2.getGaussianKernel(kernel_size, sigma)
    G = gaussian_1d @ gaussian_1d.T
    low_pass = convolve2d(low_image, G, mode='same', boundary='fill')
    blurred_high = convolve2d(high_image, G, mode='same', boundary='fill')
    high_pass = high_image - blurred_high
    hybrid = low_pass + high_weight * high_pass
    return np.clip(hybrid, 0, 255), low_pass, high_pass


def gaussian_stack(image, num_levels=5, kernel_size=21, sigma=5):
    image = image.astype(np.float64)
    gaussian_1d = cv2.getGaussianKernel(kernel_size, sigma)
    G = gaussian_1d @ gaussian_1d.T
    stack = [image]
    current = image
    for _ in range(1, num_levels):
        if current.ndim == 3:
            blurred = np.zeros_like(current)
            for c in range(current.shape[2]):
                blurred[:, :, c] = convolve2d(current[:, :, c], G, mode='same', boundary='symm')
        else:
            blurred = convolve2d(current, G, mode='same', boundary='symm')
        current = blurred
        stack.append(current)
    return stack


def laplacian_stack(gaussian_stack_levels):
    stack = []
    for i in range(len(gaussian_stack_levels) - 1):
        stack.append(gaussian_stack_levels[i] - gaussian_stack_levels[i + 1])
    stack.append(gaussian_stack_levels[-1])
    return stack

# selfie & box / dx / dy
selfie_path = root / 'input_images' / 'selfie_color.png'
selfie_arr = skio.imread(selfie_path)
R, G, B = selfie_arr[:, :, 0], selfie_arr[:, :, 1], selfie_arr[:, :, 2]
gray_selfie = (0.2989 * R + 0.5870 * G + 0.1140 * B).astype(np.uint8)
Image.fromarray(gray_selfie).save(out_root / 'selfie_gray.png')
box_kernel = np.ones((9, 9), dtype=float) / 81
box_result = convolve2d(gray_selfie, box_kernel, mode='same', boundary='fill')
Image.fromarray(np.clip(box_result, 0, 255).astype(np.uint8)).save(out_root / 'selfie_box.png')
selfie_dx = np.abs(convolution_two_loops(gray_selfie, D_x)).astype(np.uint8)
selfie_dy = np.abs(convolution_two_loops(gray_selfie, D_y)).astype(np.uint8)
Image.fromarray(selfie_dx).save(out_root / 'selfie_dx.png')
Image.fromarray(selfie_dy).save(out_root / 'selfie_dy.png')

# cameraman
cameraman_path = root / 'input_images' / 'cameraman.png'
cameraman = skio.imread(cameraman_path)
R, G, B = cameraman[:, :, 0], cameraman[:, :, 1], cameraman[:, :, 2]
cameraman_gray = (0.2989 * R + 0.5870 * G + 0.1140 * B).astype(np.uint8)
cam_dx = convolve2d(cameraman_gray, D_x, mode='same', boundary='fill')
cam_dy = convolve2d(cameraman_gray, D_y, mode='same', boundary='fill')
mag = np.sqrt(cam_dx ** 2 + cam_dy ** 2)
edge = mag > 70
Image.fromarray(np.abs(cam_dx).astype(np.uint8)).save(out_root / 'cameraman_dx.png')
Image.fromarray(np.abs(cam_dy).astype(np.uint8)).save(out_root / 'cameraman_dy.png')
Image.fromarray((mag / mag.max() * 255).astype(np.uint8)).save(out_root / 'cameraman_grad.png')
Image.fromarray((edge * 255).astype(np.uint8)).save(out_root / 'cameraman_edges.png')

# Gaussian + DoG
single_d = cv2.getGaussianKernel(5, 0)
gaussian_2d = single_d @ single_d.T
blurred = convolve2d(cameraman_gray, gaussian_2d, mode='same', boundary='fill')
blurred_dx = convolve2d(blurred, D_x, mode='same', boundary='fill')
blurred_dy = convolve2d(blurred, D_y, mode='same', boundary='fill')
blurred_mag = np.sqrt(blurred_dx ** 2 + blurred_dy ** 2)
blurred_edge = blurred_mag > 50
Image.fromarray((blurred_mag / blurred_mag.max() * 255).astype(np.uint8)).save(out_root / 'cameraman_gaussian_grad.png')
Image.fromarray((blurred_edge * 255).astype(np.uint8)).save(out_root / 'cameraman_gaussian_edges.png')
DoG_x = convolve2d(gaussian_2d, D_x, mode='full')
DoG_y = convolve2d(gaussian_2d, D_y, mode='full')
dog_dx = convolve2d(cameraman_gray, DoG_x, mode='same', boundary='fill')
dog_dy = convolve2d(cameraman_gray, DoG_y, mode='same', boundary='fill')
dog_mag = np.sqrt(dog_dx ** 2 + dog_dy ** 2)
dog_edge = dog_mag > 50
Image.fromarray((dog_mag / dog_mag.max() * 255).astype(np.uint8)).save(out_root / 'cameraman_dog_grad.png')
Image.fromarray((dog_edge * 255).astype(np.uint8)).save(out_root / 'cameraman_dog_edges.png')

# Sharpened examples
for name, img_path in [('taj_mahal', root / 'input_images' / 'taj_mahal.jpg'), ('crying_emoji', root / 'input_images' / 'crying_emoji.JPG')]:
    img = skio.imread(img_path)
    sharpened = unsharp_mask(img, kernel_size=5, sigma=0, alpha=1.0)
    Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8)).save(out_root / f'{name}_sharpened.png')

# Hybrid examples
pairs = [
    ('amy_sarveshan', root / 'input_images' / 'amy_mog.png', root / 'input_images' / 'sarveshan_mog.png'),
    ('simba_jannet', root / 'input_images' / 'simba_baby.jpg', root / 'input_images' / 'jannet_mog.jpg'),
    ('soumily_henry', root / 'input_images' / 'soumily_mog.png', root / 'input_images' / 'henry_mog.png'),
]

for name, a_path, b_path in pairs:
    img_a = skio.imread(a_path)[:, :, :3]
    img_b = skio.imread(b_path)[:, :, :3]

    if name == 'amy_sarveshan':
        img_b = img_b[130:550, 120:480]
        img_a = resize(img_a, img_b.shape[:2], preserve_range=True)
        shifted = np.zeros_like(img_b)
        shifted[:-50, :-80] = img_b[50:, 80:]
        img_b = shifted
        h, w = img_b.shape[:2]
        smaller = resize(img_b, (int(h * 0.9), int(w * 0.9)), preserve_range=True)
        resized = np.zeros_like(img_b)
        sh, sw = smaller.shape[:2]
        y = (h - sh) // 2 - 10
        x = (w - sw) // 2 + 10
        resized[y:y+sh, x:x+sw] = smaller
        img_b = resized
        img_b = resize(img_b, img_a.shape[:2], preserve_range=True)
        hybrid, low_pass, high_pass = hybrid_image(img_b, img_a, kernel_size=21, sigma=3)
    elif name == 'simba_jannet':
        h, w = img_a.shape[:2]
        smaller = resize(img_a, (int(h * 0.8), int(w * 0.8)), preserve_range=True)
        resized = np.zeros_like(img_a)
        sh, sw = smaller.shape[:2]
        y = (h - sh) // 2
        x = (w - sw) // 2
        resized[y:y+sh, x:x+sw] = smaller
        img_a = resized
        img_a = resize(img_a, img_b.shape[:2], preserve_range=True)
        shifted = np.zeros_like(img_b)
        shifted[40:, :-40] = img_b[:-40, 40:]
        img_b = shifted
        hybrid, low_pass, high_pass = hybrid_image(img_b, img_a, kernel_size=21, sigma=8, high_weight=2.5)
    else:
        h, w = img_a.shape[:2]
        henry = img_b
        y_start = (henry.shape[0] - h) // 2
        x_start = (henry.shape[1] - w) // 2
        henry = henry[y_start:y_start + h, x_start:x_start + w]
        smaller = resize(henry, (int(h * 0.70), int(w * 0.70)), preserve_range=True)
        centered = np.zeros_like(henry)
        sh, sw = smaller.shape[:2]
        x = (w - sw) // 2 + 30
        y = (h - sh) // 2 + 40
        centered[y:y+sh, x:x+sw] = smaller
        henry = centered
        hybrid, low_pass, high_pass = hybrid_image(img_a, henry, kernel_size=21, sigma=5, high_weight=1)
    Image.fromarray(np.clip(hybrid, 0, 255).astype(np.uint8)).save(out_root / f'{name}_hybrid.png')

# oraple stack
oraple = skio.imread(root / 'input_images' / 'oraple.png')
if oraple.ndim == 3 and oraple.shape[2] > 3:
    oraple = oraple[:, :, :3]

G = gaussian_stack(oraple, num_levels=5, kernel_size=21, sigma=5)
L = laplacian_stack(G)
for i, level in enumerate(G):
    img = level.copy()
    if img.max() > 1:
        img = img / 255.0
    img = np.clip(img, 0, 1)
    plt.imsave(out_root / f'oraple_gaussian_{i}.png', img)
for i, level in enumerate(L):
    max_abs = np.max(np.abs(level))
    if max_abs > 0:
        display = (level + max_abs) / (2 * max_abs)
    else:
        display = np.zeros_like(level)
    display = np.clip(display, 0, 1)
    plt.imsave(out_root / f'oraple_laplacian_{i}.png', display)

print(f'Wrote project images to {out_root}')
