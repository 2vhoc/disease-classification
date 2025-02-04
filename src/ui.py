from tkinter import *
from tkinter import ttk
import torch
from evaluation import model
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import cv2
from torchvision.transforms import ToTensor
desease = ['Eczema 1677', ' Warts Molluscum and other Viral Infections - 2103', 'Melanoma 15.75k', 'Atopic Dermatitis - 1.25k', 'Basal Cell Carcinoma (BCC) 3323', 'Melanocytic Nevi (NV) - 7970', 'Benign Keratosis-like Lesions (BKL) 2624', 'Psoriasis pictures Lichen Planus and related diseases - 2k', 'Seborrheic Keratoses and other Benign Tumors - 1.8k', 'Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k']
img: str
main = Tk()
main.geometry("1200x600")
image_frame_left = Frame(main, width=448, height=448, bg="lightgray")
image_frame_left.pack(side=LEFT)
# image_frame_left.pack_propagate(False)
image_frame_right = Frame(main, width=448, height=448, bg="lightgray")
image_frame_right.pack(side=RIGHT)
# image_frame_right.pack_propagate(False)
img_or: None
def get_file():
    global img, img_or
    path = askopenfilename()
    if path:
        img = Image.open(path)
        img = img.resize((448, 448))
        img_or = path
        img = ImageTk.PhotoImage(img)
        Label(image_frame_left, image=img).pack()
def output():
    global img_or
    imgcv = cv2.imread(img_or)
    imgcv = cv2.resize(imgcv, (224, 224))
    imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
    imgcv = ToTensor()(imgcv)
    imgcv = torch.stack([imgcv], dim=0)
    imgcv = imgcv.to('cuda')
    pred = model(imgcv)
    print(pred)
    _, i = torch.max(pred, 1)

    # i = 5
    text = f'Benh cua ban la: {desease[i]} can di chua ngay lap tuc:>'
    Label(image_frame_right, text=text).pack()
def BUTTON():
    Button(main, text="Chon anh can du doan", command=get_file).pack(pady=100)
    Button(main, text="OUTPUT", command=output).pack(pady=10)

if __name__ == "__main__":
    BUTTON()





main.mainloop()


