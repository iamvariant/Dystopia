# 6 trillion imports

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt

# image capture from webcam
width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
lmain = tk.Label(root)
lmain.pack()

# show webcam capture
def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

show_frame()
root.mainloop()

# convert capture to array
_, frame = cap.read()
frame = cv2.flip(frame, 1)
cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = Image.fromarray(cv2image)
imgtk = ImageTk.PhotoImage(image=img)




