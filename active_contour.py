import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# Initialize default parameter values
# alpha = 0.007
# beta = 0.5
# gamma = 0.004
# iterations = 5000
alpha = 0.008
beta = 40
gamma = 0.001
iterations = 10000



b = 200
n = 30
t = 3
a = 0.1
k = np.arange(230, 250, 1)
print(k)
kernel = np.ones((5, 5), dtype=np.uint8)
size = 3
structuring_element = np.ones((size, size), dtype=np.uint8)
structuring_element[size//2, size//2] = 0
size = 3
neighborhood = np.ones((size, size), dtype=np.uint8)
neighborhood[size//2, size//2] = 0 

mov_file_path = 'vidd.mp4'

def apply_snakes_model(image):
    global alpha, beta, gamma, iterations

    # Convert the image to grayscale
    gray_image = image
    height, width = gray_image.shape

    # t = np.arange(0, 2*np.pi, 0.02)
    # x = width*4.5/8+ 200*np.cos(t)
    # y = height/2+ 200*np.sin(t)
    t = np.arange(0, 2*np.pi, 0.02)
    x = width*4/8+ 70*np.cos(t)
    y = height*2/10+ 70*np.sin(t)

    init = np.array([y, x]).T

    snake = active_contour(gaussian(gray_image, 1, preserve_range=False),
                    init, alpha=alpha, beta=beta, gamma=gamma, max_num_iter=iterations)
    snake = np.array(snake, dtype=np.int32)[:, ::-1]
    snake = snake.reshape(-1, 1, 2)

    return snake

def create_empty_frame(height, width):
    blank_image = np.zeros((height, width), dtype=np.uint8)
    return blank_image

def get_contour_edge_pixels(contour_image):
    # Find contours in the contour image
    contours, _ = cv2.findContours(contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Extract the edge pixels from the contour
    edge_pixels = np.concatenate([contour for contour in contours])
    return edge_pixels

def calculate_edge_energy(contour_image, distance_image, x_e):
    S_x_e = np.argwhere(neighborhood == 1) + x_e
    #S_x_e = S_x_e.reshape(-1, 2)


    N_S_x_e = len(S_x_e)

    total = 0
    for x_ei in S_x_e:
        totalSub = 0
        for neighbor in x_ei:
            totalSub += contour_image[neighbor[1], neighbor[0]].astype(np.int32) - distance_image[neighbor[1], neighbor[0]].astype(np.int32)
        total += totalSub
    #edge_energy = np.sum((contour_image[S_x_e] - distance_image[S_x_e])) / N_S_x_e

    return total/N_S_x_e


def create_empty_frame(height, width):
    blank_image = np.zeros((height, width), dtype=np.uint8)
    return blank_image

def get_frame(video, index):
    video.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, f = video.read()
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return ret, f

def get_diff_frame(frame, prev_frame):
    O = cv2.absdiff(frame, prev_frame)
    return O

def dilate_contour_object(contour_object, diff_img):
    dilatedC = contour_object.copy()
    for ki in k:
        ret, x_inside = cv2.threshold(dilatedC, 254, 255, cv2.THRESH_BINARY)

        diff = cv2.subtract(dilatedC, diff_img, mask=x_inside)        
        ret, mask = cv2.threshold(diff, ki, 255, 
        cv2.THRESH_BINARY_INV) 
        mask = mask & x_inside
        dilated = cv2.dilate(mask, kernel, iterations=1)
        dilatedC = cv2.max(dilatedC , dilated)
    return dilatedC

def get_eroded_frame(frame):
    eroded = cv2.erode(frame, kernel, iterations=1)
    return eroded

def get_distance_image(frame):
    eroded_frame = get_eroded_frame(frame)
    diff = get_diff_frame(frame, eroded_frame)
    _, binarized_image = cv2.threshold(diff, b, 255, cv2.THRESH_BINARY)
    distance_image = binarized_image.copy()
    for _ in range(n):
        distance_image = cv2.dilate(distance_image, structuring_element)
    return distance_image
# Usage example

def get_contour_image_edge_energy(frame, contour_obj, distance_img):
    edge_pixels = get_contour_edge_pixels(contour_obj)
    Eedge = calculate_edge_energy(contour_obj, distance_img, edge_pixels)
    return Eedge

def get_contour_image_hist_energy(val, hist):
    Ehist = 0
    for ti in range(-t, t):
        value = val+ti
        if value>255:
            continue
        Ehist += hist[value]
    return Ehist

try:
    video = cv2.VideoCapture(mov_file_path)
except Exception as e:
    print(f'Error reading file: {e}')
    exit()

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frames)
frame_index = 0
#frame_index = total_frames-116

ret, prev_frame = get_frame(video, frame_index)

contour = apply_snakes_model(prev_frame)
C = create_empty_frame(prev_frame.shape[0], prev_frame.shape[1])

cv2.drawContours(C, contour, -1, (255, 255, 255), 1)
img_with_contour = prev_frame.copy()
img_with_contour = cv2.max(C , img_with_contour)

cv2.imshow('First Frame with Contour', img_with_contour)
if cv2.waitKey(0) & 0xFF == ord('q'):
    pass

for i in range(frame_index+1, total_frames):
#for i in range(frame_index-1, -1, -1):
    ret, frame = get_frame(video, i)
    O = get_diff_frame(frame, prev_frame)

    dilatedC = dilate_contour_object(C, O)

    dilatedC_edge_pixels = get_contour_edge_pixels(dilatedC)

    print(frame.shape)
    distance_image = get_distance_image(frame)
    hist = cv2.calcHist([prev_frame], [0], C, [256], [0, 256])
    hist_star = cv2.calcHist([frame], [0], dilatedC, [256], [0, 256])

    print("Begin erosion in contour pixels: " + str(dilatedC_edge_pixels.shape[0]))
    for ep in dilatedC_edge_pixels:
        ep = ep[0]
        ep_x = ep[0]
        ep_y = ep[1]

        Ehist = get_contour_image_hist_energy(prev_frame[ep_y, ep_x], hist)
        Ehist_star = get_contour_image_hist_energy(frame[ep_y, ep_x], hist_star)

        neighboring_elements = np.argwhere(neighborhood == 1)[:, [1,0]]
        neighboring_elements[:, 1] += ep_y
        neighboring_elements[:, 0] += ep_x

        Eedge = calculate_edge_energy(dilatedC, distance_image, ep.reshape(-1,1,2))

        og = dilatedC[neighboring_elements[:,1], neighboring_elements[:,0]]

        dilatedC[neighboring_elements[:,1], neighboring_elements[:,0]] = np.min(og)
        Eedge_star = calculate_edge_energy(dilatedC, distance_image, ep.reshape(-1,1,2))
        if (Ehist_star/Ehist)>1.0 and (Eedge_star/Eedge) < (1.0+a):
            pass
        else:
            dilatedC[neighboring_elements[:,1], neighboring_elements[:,0]]=og

    C= dilatedC

    cont, _ = cv2.findContours(C, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #C = create_empty_frame(prev_frame.shape[0], prev_frame.shape[1])
    #cv2.drawContours(C, cont, -1, (255, 255, 255), 1)

    img_with_contour = frame.copy()
    #img_with_contour = cv2.drawContours(img_with_contour, cont, -1,(255, 255, 255), 2)

    img_with_contour = cv2.max(C , img_with_contour)

    cv2.imshow('First Frame with Contour', img_with_contour)

    prev_frame = frame
    if cv2.waitKey(0) & 0xFF == ord('q'):
        pass

video.release()
cv2.destroyAllWindows()