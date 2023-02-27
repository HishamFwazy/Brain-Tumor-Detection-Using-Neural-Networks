from tkinter import *
import cv2
import imutils
import numpy as np
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from os import listdir
from sklearn.utils import shuffle





class ND():
    def ImageProcessing(self,filename):
        f_types = [('Jpg Files', '*.jpg'), ('PNG Files', '*.png')]  # type of files to select
        filename = filedialog.askopenfilename(multiple=True, filetypes=f_types)
        #col = 1  # start from column 1
        #row = 4  # start from row 3
        for f in filename:
            '''
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
            '''
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
    def load_data( self,dir_list, image_size):
        """
        Read images, resize and normalize them.
        Arguments:
            dir_list: list of strings representing file directories.
        Returns:
            X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
            y: A numpy array with shape = (#_examples, 1)
        """

        # load all images in a directory
        X = []
        y = []
        image_width, image_height = image_size

        for directory in dir_list:
            for filename in listdir(directory):
                # load the image
                image = cv2.imread(directory + '\\' + filename)
                # crop the brain and ignore the unnecessary rest part of the image
                image = lambda:[self.ImageProcessing()]
                # resize image
                image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
                # normalize values
                image = image / 255.
                # convert image to numpy array and append it to X
                X.append(image)
                # append a value of 1 to the target array if the image
                # is in the folder named 'yes', otherwise append 0.
                if directory[-3:] == 'yes':
                    y.append([1])
                else:
                    y.append([0])

        X = np.array(X)
        y = np.array(y)

        # Shuffle the data
        X, y = shuffle(X, y)

        print(f'Number of examples is: {len(X)}')
        print(f'X shape is: {X.shape}')
        print(f'y shape is: {y.shape}')

        return X, y



    def split_data(self,X, y, test_size=0.2):
        """
        Splits data into training, development and test sets.
        Arguments:
            X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
            y: A numpy array with shape = (#_examples, 1)
        Returns:
            X_train: A numpy array with shape = (#_train_examples, image_width, image_height, #_channels)
            y_train: A numpy array with shape = (#_train_examples, 1)
            X_val: A numpy array with shape = (#_val_examples, image_width, image_height, #_channels)
            y_val: A numpy array with shape = (#_val_examples, 1)
            X_test: A numpy array with shape = (#_test_examples, image_width, image_height, #_channels)
            y_test: A numpy array with shape = (#_test_examples, 1)
        """

        X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
        X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)

        return X_train, y_train, X_val, y_val, X_test, y_test









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

        X, y = self.load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))
        X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(X, y, test_size=0.3)

        U_b1 = Button(newWindow, text='Upload Image to Convert', font=my_font1, bg="gray90",height=3, width=22, command=lambda:[self.ImageProcessing()])
        U_b1.grid(row=1,column=1,columnspan=4)




        #Detect_button = Button(newWindow, text="Tumor Detection", font=my_font1,bg="gray90",height=3, width=20,command=self.tumor_file() )

        #Detect_button.place(x=715 ,y =1)

        #grid(row=2, column=32, columnspan=4)


        newWindow.mainloop()




ND1=ND()

