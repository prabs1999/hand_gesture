from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import subprocess
import keyboard


window = Tk()
window.title("Human Computer Interaction using Hand gesture Recognition")
window.geometry('1200x650')

#window.configure(image = bg)
bg = ImageTk.PhotoImage(Image.open(r"C:\Users\SUSHANT\Downloads\1614604.jpg")) 
canvas1 = Canvas(window, width = 1200, height = 650)
canvas1.pack(fill = "both", expand = True)
canvas1.create_image( 0, 0, anchor = "nw",image = bg )

heading = Label(window, text="Enter the following details", bg="grey", font = ("Times New Roman", 20, "bold"))
canvas1.create_window(600, 100, window=heading)

app_label =Label(window, text="Select the application : ", bg="grey", font = ("Times New Roman", 15, "bold"))
canvas1.create_window(300, 175, window=app_label)

app = ('Power Point','MS WORD')
n = StringVar()
app_choosen = ttk.Combobox(window, width = 27, 
                            textvariable = n)
app_choosen['values'] = app
canvas1.create_window(600, 175, window=app_choosen)
app_choosen.focus_set()
ftype = ""
def callback(ask):
    global ftype, actions_list
    ftype = app[app_choosen.current()]
    if (ftype == "Power Point"):
        actions_list = ppt_list
    elif (ftype == "MS WORD"):
        actions_list = word_list
    button_explore.focus_set()
    
app_choosen.bind("<<ComboboxSelected>>", callback)
file_label = Label(window, text="Select the file              : ", bg="grey", font = ("Times New Roman", 15, "bold"))
canvas1.create_window(300, 225, window=file_label)


label_file_explorer = Label(window, text = "Choose your file",width = 20, bg="white", font = ("Times New Roman", 15, "bold"))
canvas1.create_window(600, 225, window=label_file_explorer)

def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("All files",
                                                        "*.*"),("Text files",
                                                        "*.txt*")))
      
    # Change label contents
    label_file_explorer.configure(text = filename)
    actions_1['values'] = actions_list
    actions_1.focus_set()
    
button_explore = Button(window, text = "Browse Files", command = browseFiles)
canvas1.create_window(775, 225, window=button_explore)

gesture_label = Label(window, text="Select an action for each gesture : ", bg="grey", font = ("Times New Roman", 15, "bold"))
canvas1.create_window(300, 275, window=gesture_label)

palm_label = Label(window, text="Palm      : ", bg="grey", font = ("Times New Roman", 15, "bold"))
canvas1.create_window(300, 325, window=palm_label)
peace_label = Label(window, text="Peace    : ", bg="grey", font = ("Times New Roman", 15, "bold"))
canvas1.create_window(300, 375, window=peace_label)
thumbs_label = Label(window, text="Thumbs Up : ", bg="grey", font = ("Times New Roman", 15, "bold"))
canvas1.create_window(300, 425, window=thumbs_label)

ppt_list = ['Open','Slideshow','Close']
word_list = ['Open','Scroll','Close']
actions_list = []



actions_1 = ttk.Combobox(window, width = 27, 
                            textvariable = StringVar())

actions_2 = ttk.Combobox(window, width = 27, 
                            textvariable = StringVar())

actions_3 = ttk.Combobox(window, width = 27, 
                            textvariable = StringVar())


canvas1.create_window(600, 325, window=actions_1)
canvas1.create_window(600, 375, window=actions_2)
canvas1.create_window(600, 425, window=actions_3)

gest_act = {}

def select_action(x):
    global gest_act
    gest_act["Palm"] = actions_list[actions_1.current()]
    actions_list.pop(actions_1.current())
    actions_2['values'] = actions_list
    actions_2.focus_set()

def select_action_1(x):
    global gest_act
    gest_act["Peace"] = actions_list[actions_2.current()]
    actions_list.pop(actions_2.current())
    actions_3['values'] = actions_list
    actions_3.focus_set()
    
def select_action_2(x): 
    global gest_act
    gest_act["Thumbs Up"] = actions_list[actions_3.current()]
    submit_button.focus_set()
    
actions_1.bind("<<ComboboxSelected>>", select_action)
actions_2.bind("<<ComboboxSelected>>", select_action_1)
actions_3.bind("<<ComboboxSelected>>", select_action_2)

def change_canvas():
    canvas1.pack_forget()
    canvas2.pack(fill = "both", expand = True)
    
submit_button = Button(window, text = "Submit", command = change_canvas)
canvas1.create_window(600, 475, window=submit_button)

model = keras.models.load_model(r"C:\Users\SUSHANT\.spyder-py3\HCI_model.h5")
background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 155
ROI_left = 350





word_dict = {0:"Nothing", 1:'Palm',2:'Peace',3:'Thumbs Up'}
word_weight = {"Nothing":0,'Palm':0, 'Peace': 0, 'Thumbs Up': 0}

def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)
    
def segment_hand(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    #Fetching contours in the frame (These contours can be of hand or any other object in foreground) ...
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand 
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        # Returning the hand segment(max contour) and the thresholded image of hand...
        return (thresholded, hand_segment_max_cont)


cam = cv2.VideoCapture(0)
num_frames =0
prev_label = ""

def main_fun():
    ret, frame = cam.read()
    
    # filpping the frame to prevent inverted image of captured frame...
    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    # ROI from the frame
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
    global num_frames
    global word_weight
    global word_dict
    global prev_label

    if num_frames < 70:
        
        cal_accum_avg(gray_frame, accumulated_weight)
        
        cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    
    else: 
        # segmenting the hand region
        hand = segment_hand(gray_frame)
        

        # Checking if we are able to detect the hand...
        if hand is not None:
            
            thresholded, hand_segment = hand

            # Drawing contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0),1)
            
            img_2 = Image.fromarray(thresholded)
            imgtk_1 = ImageTk.PhotoImage(image=img_2)
            lmain_1.imgtk = imgtk_1
            lmain_1.configure(image=imgtk_1)
            #cv2.imshow("Thesholded Hand Image", thresholded)
            
            thresholded = cv2.resize(thresholded, (64, 64))
            thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
            thresholded = np.reshape(thresholded, (1,thresholded.shape[0],thresholded.shape[1],3))
            
            pred = model.predict(thresholded)
            if (num_frames % 40) != 0:
                word_weight[word_dict[np.argmax(pred)]] = word_weight[word_dict[np.argmax(pred)]] + 1
            else:
                label = list(word_weight.keys())[list(word_weight.values()).index(max(word_weight.values()))] 
                cv2.putText(frame_copy, label, (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                if (label != prev_label):
                    prev_label = label
                    ac = gest_act[label]
                    
                    if ac == 'Open':
                        os.startfile(label_file_explorer.cget("text"))
                    elif ac == 'Slideshow':
                        keyboard.press_and_release('F5')
                    elif ac == 'Scroll':
                        keyboard.press_and_release("Page Down")
                    elif ac == 'Close':
                        os.system('wmic process where name="POWERPNT.exe" delete')
                        os.system('wmic process where name="WINWORD.exe" delete')
                
                word_weight = {"Nothing":0,'Palm':0, 'Peace': 0, 'Thumbs Up': 0}
            
    # Draw ROI on frame_copy
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)

    # incrementing the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand
    cv2.putText(frame_copy, "DataFlair hand sign recognition_ _ _", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
    frame_copy=cv2.resize(frame_copy,(600,400))
    
    img_1 = Image.fromarray(frame_copy)
    imgtk = ImageTk.PhotoImage(image=img_1)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    
    
    k = cv2.waitKey(1) & 0xFF
    
    if keyboard.is_pressed("esc"):
        cam.release()
        cv2.destroyAllWindows()
        return
    
    lmain.after(1, main_fun)


canvas2 = Canvas(window, width = 1200, height = 650)
canvas2.create_image( 0, 0, anchor = "nw",image = bg)


def start_cam():
    main_fun()
        

button1 = Button(window, text = "Start", command = start_cam) 
canvas2.create_window(600, 100, window = button1)

frame = Frame(window, bg = "white", height = 400, width = 600)
lmain = Label(frame, bg = "white")
lmain.grid()
canvas2.create_window(350, 350, window = frame)

frame_1 = Frame(window, bg = "white", height = 200, width = 200)
lmain_1 = Label(frame_1, bg = "white")
lmain_1.grid()
canvas2.create_window(900, 350, window = frame_1)
    


window.mainloop()
