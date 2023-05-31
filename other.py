import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import tkinter as tk
from tkinter import Scale, Label, Button, Entry

# Initialize default parameter values
alpha = 0.04
beta = 30
gamma = 0.01
iterations = 5000
frame = []
def apply_snakes_model(image):
    global alpha, beta, gamma, iterations

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape

    t = np.arange(0, 2*np.pi, 0.02)
    x = width*4/8+ 70*np.cos(t)
    y = height*2/10+ 70*np.sin(t)

    init = np.array([y, x]).T

    snake = active_contour(gaussian(gray_image, 1, preserve_range=False),
                    init, alpha=alpha, beta=beta, gamma=gamma, max_num_iter=iterations)
    return snake
def read_mov_file(file_path):
    global frame

    try:
        video = cv2.VideoCapture(file_path)
    except Exception as e:
        print(f'Error reading file: {e}')
        return
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = total_frames-116
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = video.read()
    print(frame.shape)

    def update_parameters():
        global alpha, beta, gamma, iterations
        alpha = float(alpha_entry.get())
        beta = float(beta_entry.get())
        gamma = float(gamma_entry.get())
        iterations = int(iterations_entry.get())

    def process_frame():
        global frame
        # Apply the snakes model to the first frame
        frame_with_contour = frame.copy()

        contour = apply_snakes_model(frame_with_contour)
        contour = np.array(contour, dtype=np.int32)[:, ::-1]
        print(contour.shape)
        contour = contour.reshape(-1, 1, 2)

        # Draw the contour on the frame
        #frame_with_contour = frame.copy()
        cv2.drawContours(frame_with_contour, contour, -1, (255, 255, 255), 1)

        # Display the resulting frame with the contour
        cv2.imshow('First Frame with Contour', frame_with_contour)
    def process_and_update():
        update_parameters()
        process_frame()

    print("wa2r")

    # Create the UI
    root = tk.Tk()
    root.title('Snake Model Parameters')
    print("war")
    # Alpha parameter
    alpha_label = Label(root, text='Alpha')
    alpha_label.pack()
    alpha_entry = Entry(root)
    alpha_entry.insert(0, str(alpha))
    alpha_entry.pack()

    # Beta parameter
    beta_label = Label(root, text='Beta')
    beta_label.pack()
    beta_entry = Entry(root)
    beta_entry.insert(0, str(beta))
    beta_entry.pack()

    # Gamma parameter
    gamma_label = Label(root, text='Gamma')
    gamma_label.pack()
    gamma_entry = Entry(root)
    gamma_entry.insert(0, str(gamma))
    gamma_entry.pack()

    # Iterations parameter
    iterations_label = Label(root, text='Iterations')
    iterations_label.pack()
    iterations_entry = Entry(root)
    iterations_entry.insert(0, str(iterations))
    iterations_entry.pack()

    # Process button
    process_button = Button(root, text='Process Frame', command=process_and_update)
    process_button.pack()
    root.mainloop()
    if cv2.waitKey(0) & 0xFF == ord('q'):
        return

    video.release()
    cv2.destroyAllWindows()

# Usage example
mov_file_path = 'vidd.mp4'
read_mov_file(mov_file_path)
