from tkinter import *
import NewDetection
from PIL import ImageTk
import tkinter.messagebox as mymessagebox

class HP():
    def __init__(self):

        # window
        newWindow = Tk()
        newWindow.geometry('550x560')
        newWindow.resizable(False, False)
        newWindow.iconbitmap(r'E:\SSD\Uni\Graduation Project\GUI\icons8-homepage-64.ico')
        newWindow.title('Home Page')
        newWindow.configure(background='white')

        # Frame

        Frame_hp = Frame(newWindow, bg="gray90")
        Frame_hp.place(x=20, y=20, height=500, width=500)
        bg = ImageTk.PhotoImage(file=r"""E:\SSD\Uni\Graduation Project\Images\GettyImages611992272Tiny.jpg""")
        bg_image = Label(Frame_hp, image=bg).place(x=-15, y=0, relheight=2, relwidth=1.1)

        #menubar
        menubar=Menu(newWindow)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.exit)
        menubar.add_cascade(label="File", menu=filemenu)
        helpmenu = Menu(menubar, tearoff=0)
        #helpmenu.add_command(label="Help Index", command=self.donothing)
        helpmenu.add_command(label="About...", command=self.donothing)
        menubar.add_cascade(label="Help", menu=helpmenu)
        newWindow.config(menu=menubar)

        # buttons
        history_button = Button(Frame_hp, text="HISTORY",height=3, width=10 ,bg="lightsteelblue2",font=("cambria", 15, "bold"),borderwidth=5).place(x=10, y=200)
        N = NewDetection.ND

        ndetection_button = Button(Frame_hp, text="Tumor Detection",height=3, width=15 ,command=lambda:[newWindow.destroy(),N()], bg="lightsteelblue2",font=("cambria", 15, "bold"),borderwidth=5).place(x=300, y=200)
        #newWindow.destroy()
        newWindow.mainloop()

    def exit(self):
        exit()
    def donothing(self):
        mymessagebox.showinfo("About","Brain Tumor Detection Application\n\t2022")







#HP1=HP()