from tkinter import *
import HomePage
from PIL import ImageTk
import tkinter.messagebox as mymessagebox
import pyodbc




def validateLogin(*event):

    conn = pyodbc.connect('Driver={SQL Server};'
                          'Server=KIMOO\SQLEXPRESS;'
                          'Database=Brain Tumor Detection using CNN;'
                          'Trusted_Connection=yes;')

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM LoginForm where username = '" + username.get() + "' and password = '" + password.get() + "';")
    myresult = cursor.fetchone()
    if myresult == None:
        mymessagebox.showerror("Error", "Invalid User Name or Password")

    else:

         tkWindow.destroy()
         HomePage.HP()
         conn.close()
         cursor.close()

#window



tkWindow = Tk()
tkWindow.geometry('1300x600+200+50')
tkWindow.resizable(False,False)
tkWindow.iconbitmap(r"""E:\SSD\Uni\Graduation Project\GUI\download.ico""")
tkWindow.title('LOGIN')


bg=ImageTk.PhotoImage(file=r"""E:\SSD\Uni\Graduation Project\GUI\brain1.jpg""")
bg_image=Label(tkWindow,image=bg).place(x=0,y=0,relheight=1,relwidth=1)

#LoginLabel= Label(tkWindow , text= "Login" , font=myFont).grid(row=0 , column=5)
Frame_login = Frame(tkWindow, bg="gray90")
Frame_login.place(x=20, y=20, height=500, width=500)
title = Label(Frame_login, text="Login", font=("cambria", 35, "bold"), bg="gray90").place(x=180, y=30)

#username label and text entry box
lbl_user=Label(Frame_login,text="Username",font=("cambria",15,"bold"),bg="gray90").place(x=30,y=130)
username = StringVar()
usernameEntry = Entry(Frame_login, textvariable=username,font=("times new roman",15),bg="gray100")
usernameEntry.place(x=30,y=170,width=350,height=35)

#password label and password entry box
lbl_pass = Label(Frame_login, text="Password", font=("cambria", 15, "bold"), bg="gray90").place(x=30, y=230)
password = StringVar()
passwordEntry = Entry(Frame_login, textvariable=password,  font=("times new roman", 15), bg="gray100",show='*')
passwordEntry.place(x=30, y=270, width=350, height=35)


#login button
login_button=Button(Frame_login,text="Login",bg="gray90",command=validateLogin,font=("cambria",15,"bold")).place(x=30,y=350)
tkWindow.bind('<Return>',validateLogin)

#tkWindow.configure(background='#54596d')


'--------------------|||----------------------------''--------------------|||----------------------------'

'--------------------|||----------------------------''--------------------|||----------------------------'

'--------------------|||----------------------------''--------------------|||----------------------------'


tkWindow.mainloop()





