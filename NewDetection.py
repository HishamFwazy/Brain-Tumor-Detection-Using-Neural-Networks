from tkinter import *
import cv2
import imutils
import numpy as np
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from os import listdir
from sklearn.utils import shuffle





class ND():
    def ImageProcessing(self):

        f_types = [('Jpg Files', '*.jpg'), ('PNG Files', '*.png')]  # type of files to select
        filename = filedialog.askopenfilename(multiple=True, filetypes=f_types)
        originalImage = cv2.imread(filename)
        #col = 1  # start from column 1
        #row = 4  # start from row 3
        '''
        for f in filename:
            
            img = Image.open(f)  # read the image file
            img = img.resize((300, 300))  # new width & height
            img = ImageTk.PhotoImage(img)
            e1 = Label()
            e1.grid(row=row, column=col)
            e1.image = img
            e1['image'] = img  # garbage collection
            #if (col == 3):  # start new line after third column
                #row = row + 1  # start wtih next row
                #col = 1  # start with first column
            #else:  # within the same row
               # col = col + 1  # increase to next column
            
            originalImage = cv2.imread(f)

            # Convert the image to grayscale, and blur it slightly
            grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
            grayImage = cv2.GaussianBlur(grayImage, (5, 5), 0)

            # Convert to black and white
            (thresh, blackAndWhiteImage) = cv2.threshold(originalImage, 127, 255, cv2.THRESH_BINARY)

            # Threshold the image, then perform a series of erosions +
            # dilations to remove any small regions of noise
            thresh = cv2.threshold(grayImage, 45, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours in thresholded image, then grab the largest one
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)

            # Find the extreme points
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            # crop new image out of the original image using the four extreme points (left, right, top, bottom)
            new_image = blackAndWhiteImage[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

            #Output Image
            cv2.imshow('Original image', originalImage)
            cv2.imshow('Black&White image', blackAndWhiteImage)
            cv2.imshow('Cropped image', new_image)
         '''

    def __init__(self):


    #window
        newWindow = Tk()
        newWindow.geometry('310x100+200+50')
        newWindow.resizable(False, False)
        #newWindow.iconbitmap(r'E:\SSD\Uni\Graduation Project\GUI\icons8-homepage-64.ico')
        newWindow.title('Tumor Detection')
        newWindow.configure(background='white')
        my_font1 = ('cambria', 18, 'bold')

        augmented_path = r'E:\SSD\Uni\Graduation Project\Brain-Tumor-Detection-master\augmented data/'

        # augmented data (yes and no) contains both the original and the new generated examples
        augmented_yes = augmented_path + 'yes'
        augmented_no = augmented_path + 'no'

        IMG_WIDTH, IMG_HEIGHT = (240, 240)



        U_b1 = Button(newWindow, text='Upload Image to Convert', font=my_font1, bg="gray90",height=3, width=22, command=lambda:[self.ImageProcessing()])
        U_b1.grid(row=1,column=1,columnspan=4)




        #Detect_button = Button(newWindow, text="Tumor Detection", font=my_font1,bg="gray90",height=3, width=20,command=self.tumor_file() )

        #Detect_button.place(x=715 ,y =1)

        #grid(row=2, column=32, columnspan=4)


        newWindow.mainloop()




ND1=ND()

