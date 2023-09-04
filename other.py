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

torch = []
model = []
data = []

# Check if we can perform gpu computing
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = 'cpu'


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, 
                            num_workers=4, 
                            pin_memory=True, 
                            prefetch_factor= 4)


test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor= 4)


import torch.nn as nn
# Send the model to GPU memory
model.to(device)
# Wrap the model with DataParallel
model = nn.DataParallel(model)


# Send data to GPU memory
data.to(device)
GradScaler = []
epochs =[]
optimizer =[]
autocast = []
loss_fn=[]


# Creates a GradScaler once at the beginning of training.
scaler = torch.cuda.amp.GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        # Runs the forward pass with autocasting.
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        # Updates the scale for next iteration.
        scaler.update()
DATASET_ID='dasd'

dataset = load_dataset("path/to/local/loading_script", split="train")  # equivalent because the file has the same name as the directory
dataset = load_dataset("path/to/local/loading_script", split="train[30:]")  # equivalent because the file has the same name as the directory
dataset = load_dataset("path/to/local/loading_script", split="train[10%:30%]")  # equivalent because the file has the same name as the directory
dynamic_transform = []
static_transform = []

from datasets import load_dataset

dataset = load_dataset(DATASET_ID)

dataset.map(static_transform, batched=True, batch_size=4096, num_proc=16)

dataset.set_transform(dynamic_transform)


from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

def main():
    dist.init_process_group("nccl") # 'nccl' for GPU - 'gloo' for CPU
    rank = dist.get_rank() # Device ID
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])# Local Device ID
    world_size = int(os.environ['WORLD_SIZE'])# Total device number
    master_addr = os.environ['MASTER_ADDR']

    # // model, dataset, dataloader init

    model.to(rank)
    model = DDP(
        model,
        device_ids=[rank]
    )

    # // train/test loop

if __name__ == "__main__":
    main()



from datasets import Dataset
import torch.distributed

dataset = load_dataset(DATASET_ID, split="train")

if rank > 0:
    torch.distributed.barrier()

dataset2 = dataset.map(static_transform)

if rank == 0:
    torch.distributed.barrier()

corr = torch.Tensor([correct]).to(rank)
tot = torch.Tensor([total]).to(rank)
dist.all_reduce(corr, op=dist.ReduceOp.SUM)
dist.all_reduce(tot, op=dist.ReduceOp.SUM)


def static_transform(examples):
    examples["pixel_values"] = [image.convert("RGB").resize((32, 32)) for image in examples["image"]]
    return examples

def dynamic_transform(examples):
    examples["pixel_values"] = [transform(image) for image in examples["pixel_values"]]
    return examples



from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(dataset=dataset, shuffle=True)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=train_sampler, num_workers=10)

test_sampler = DistributedSampler(dataset=dataset, shuffle=False)
test_loader  = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=test_sampler, num_workers=10)


# Training loop
for epoch in tqdm(range(10)):
    train_loader.sampler.set_epoch(epoch)
    for i, data in tqdm(enumerate(train_loader), leave=False):