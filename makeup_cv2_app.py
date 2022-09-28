from typing import Text
from kivy.app import App
from kivy.core import image, window
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.core.window import Window 
from kivy.graphics.texture import Texture
from kivy.uix.textinput import TextInput

from imutils import face_utils

import cv2
import dlib
import numpy as np
import csv

predictor_68 = "/home/bearcide/Desktop/code/makeup_cv2_app/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_68)

flag_crop = False
flag_crop_type = 0
flag_makeup = False
flag_makeup_type = 0

y_offseth = ""
x_offseth = ""

r_input = ""
g_input = ""
b_input = ""
a_input = ""

save = False
wipe = False
name_save = ""
type_save = ""

class CamApp(App):
    def build(self):
        Window.size = (1000, 480) #la webcam es 480p
        Window.clearcolor = (.2, .4, .5, 1)
        self.img1=Image(pos = (-180, 0))
#####################   Crop Layout     #####################################################################
        button_crop_righteye = Button(text='Right eye', font_size=14, size_hint = (.1, .1), pos=(650, 410)) #650-900, 0-
        button_crop_righteye.bind(on_press = self.crop_options_righteye)
        button_crop_lefteye = Button(text='Left eye', font_size=14, size_hint = (.1, .1), pos=(750, 410)) #650-900, 0-
        button_crop_lefteye.bind(on_press = self.crop_options_lefteye)
        button_crop_mouth = Button(text='Mouth', font_size=14, size_hint = (.1, .1), pos=(850, 410)) #650-900, 0-
        button_crop_mouth.bind(on_press = self.crop_options_mouth)
        button_crop_leftbrow = Button(text='Left Eyebrow', font_size=14, size_hint = (.1, .1), pos=(650, 360)) #650-900, 0-
        button_crop_leftbrow.bind(on_press = self.crop_options_leftbrow)
        button_crop_rightbrow = Button(text='Right Eyebrow', font_size=14, size_hint = (.1, .1), pos=(750, 360)) #650-900, 0-
        button_crop_rightbrow.bind(on_press = self.crop_options_rightbrow)
        button_crop_unibrow = Button(text='Unibrow', font_size=14, size_hint = (.1, .1), pos=(850, 360)) #650-900, 0-
        button_crop_unibrow.bind(on_press = self.crop_options_unibrow)
        button_crop_leftcheeck = Button(text='Left Cheeck', font_size=14, size_hint = (.1, .1), pos=(650, 310)) #650-900, 0-
        button_crop_leftcheeck.bind(on_press = self.crop_options_leftcheeck)
        button_crop_rightcheeck = Button(text='Right Cheeck', font_size=14, size_hint = (.1, .1), pos=(750, 310)) #650-900, 0-
        button_crop_rightcheeck.bind(on_press = self.crop_options_rightcheeck)
        button_crop_jaw = Button(text='Jaw', font_size=14, size_hint = (.1, .1), pos=(850, 310)) #650-900, 0-
        button_crop_jaw.bind(on_press = self.crop_options_jaw)
#####################   Makeup Layout   #####################################################################
        button_makeup_mouth = Button(text='Mouth makeup', font_size=14, size_hint = (.1, .1), pos=(650, 180))
        button_makeup_mouth.bind(on_press = self.makeup_options_lips)
        button_makeup_eyes = Button(text='Eyeliner', font_size=14, size_hint = (.1, .1), pos=(750, 180))
        button_makeup_eyes.bind(on_press = self.makeup_options_eyes)
        button_makeup_brows = Button(text='Brows', font_size=14, size_hint = (.1, .1), pos=(850, 180))
        button_makeup_brows.bind(on_press = self.makeup_options_brows)
        button_makeup_shadow = Button(text='Eyeshadow', font_size=14, size_hint = (.1, .1), pos=(750, 130))
        button_makeup_shadow.bind(on_press = self.makeup_options_shadow)
        button_makeup_delinated = Button(text='Delinated Brows', font_size=13, size_hint = (.1, .1), pos=(850, 130))
        button_makeup_delinated.bind(on_press = self.makeup_options_delinated)
        button_makeup_facepaint = Button(text='Face Paint', font_size=13, size_hint = (.1, .1), pos=(650, 130))
        button_makeup_facepaint.bind(on_press = self.makeup_options_facepaint)

        button_makeup_save = Button(text='Save?', font_size=14, size_hint = (.1, .1), pos=(680, 45))
        button_makeup_save.bind(on_press = self.makeup_options_save)
        button_makeup_wipe = Button(text='Wipe', font_size=14, size_hint = (.1, .1), pos=(780, 45))
        button_makeup_wipe.bind(on_press = self.makeup_options_wipe)


        #General text
        text_crop = Label(text="Crop Options: Choose one", pos= (230, 230))
        text_makeup = Label(text="Makeup Options: Choose to your hearts content", pos= (305, 0))
        #Offseth options
        text_y_offseth = Label(text="The X offseth:", pos= (195, 55))
        text_x_offseth = Label(text="The Y offseth:", pos= (195, 25))
        self.y_offseth = TextInput(text="20",size_hint = (.05, .06), pos=(750, 280), multiline=False)
        self.x_offseth = TextInput(text="20",size_hint = (.05, .06), pos=(750, 250), multiline=False)
        #rgb options
        self.r_box = TextInput(text="0",size_hint = (.05, .06), pos=(730, 95), multiline=False)
        self.g_box = TextInput(text="0",size_hint = (.05, .06), pos=(780, 95), multiline=False)
        self.b_box = TextInput(text="0",size_hint = (.05, .06), pos=(830, 95), multiline=False)
        self.a_box = TextInput(text="0",size_hint = (.05, .06), pos=(880, 95), multiline=False)

        self.save_box = TextInput(text="Name",size_hint = (.08, .06), pos=(650, 95), multiline=False)
##################### Add elements to the app ####################################################################
        layout = FloatLayout()
        layout.add_widget(self.r_box)
        layout.add_widget(self.g_box)
        layout.add_widget(self.b_box)
        layout.add_widget(self.a_box)
        layout.add_widget(self.save_box)
        layout.add_widget(self.y_offseth)
        layout.add_widget(self.x_offseth)
        layout.add_widget(self.img1)
        layout.add_widget(text_crop)
        layout.add_widget(text_makeup)
        layout.add_widget(text_y_offseth)
        layout.add_widget(text_x_offseth)

        layout.add_widget(button_crop_righteye)
        layout.add_widget(button_crop_lefteye)
        layout.add_widget(button_crop_mouth)
        layout.add_widget(button_crop_leftbrow)
        layout.add_widget(button_crop_rightbrow)
        layout.add_widget(button_crop_unibrow)
        layout.add_widget(button_crop_leftcheeck)
        layout.add_widget(button_crop_rightcheeck)
        layout.add_widget(button_crop_jaw)
        
        layout.add_widget(button_makeup_mouth)
        layout.add_widget(button_makeup_eyes)
        layout.add_widget(button_makeup_brows)
        layout.add_widget(button_makeup_shadow)
        layout.add_widget(button_makeup_delinated)
        layout.add_widget(button_makeup_facepaint)

        layout.add_widget(button_makeup_save)
        layout.add_widget(button_makeup_wipe)
        #opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 0.5/30.0)
        return layout
#####################   Crop Buttons    #####################################################################     
    def crop_options_righteye(self, event):
        global flag_crop
        global flag_crop_type
        if flag_crop == False:
            flag_crop = True
            flag_crop_type = 1
        else:
            flag_crop = False
            flag_crop_type = 0
        global x_offseth
        global y_offseth
        x_offseth = self.y_offseth.text
        y_offseth = self.x_offseth.text
        

        

    def crop_options_lefteye(self, event):
        global flag_crop
        global flag_crop_type
        if flag_crop == False:
            flag_crop = True
            flag_crop_type = 2
        else:
            flag_crop = False
            flag_crop_type = 0
        global x_offseth
        global y_offseth
        x_offseth = self.y_offseth.text
        y_offseth = self.x_offseth.text

    def crop_options_mouth(self, event):
        global flag_crop
        global flag_crop_type
        if flag_crop == False:
            flag_crop = True
            flag_crop_type = 3
        else:
            flag_crop = False
            flag_crop_type = 0
        global x_offseth
        global y_offseth
        x_offseth = self.y_offseth.text
        y_offseth = self.x_offseth.text

    def crop_options_leftbrow(self, event):
        global flag_crop
        global flag_crop_type
        if flag_crop == False:
            flag_crop = True
            flag_crop_type = 4
        else:
            flag_crop = False
            flag_crop_type = 0
        global x_offseth
        global y_offseth
        x_offseth = self.y_offseth.text
        y_offseth = self.x_offseth.text

    def crop_options_rightbrow(self, event):
        global flag_crop
        global flag_crop_type
        if flag_crop == False:
            flag_crop = True
            flag_crop_type = 5
        else:
            flag_crop = False
            flag_crop_type = 0
        global x_offseth
        global y_offseth
        x_offseth = self.y_offseth.text
        y_offseth = self.x_offseth.text

    def crop_options_unibrow(self, event):
        global flag_crop
        global flag_crop_type
        if flag_crop == False:
            flag_crop = True
            flag_crop_type = 6
        else:
            flag_crop = False
            flag_crop_type = 0
        global x_offseth
        global y_offseth
        x_offseth = self.y_offseth.text
        y_offseth = self.x_offseth.text
    
    def crop_options_leftcheeck(self, event):
        global flag_crop
        global flag_crop_type
        if flag_crop == False:
            flag_crop = True
            flag_crop_type = 7
        else:
            flag_crop = False
            flag_crop_type = 0
        global x_offseth
        global y_offseth
        x_offseth = self.y_offseth.text
        y_offseth = self.x_offseth.text
    
    def crop_options_rightcheeck(self, event):
        global flag_crop
        global flag_crop_type
        if flag_crop == False:
            flag_crop = True
            flag_crop_type = 8
        else:
            flag_crop = False
            flag_crop_type = 0
        global x_offseth
        global y_offseth
        x_offseth = self.y_offseth.text
        y_offseth = self.x_offseth.text

    def crop_options_jaw(self, event):
        global flag_crop
        global flag_crop_type
        if flag_crop == False:
            flag_crop = True
            flag_crop_type = 9
        else:
            flag_crop = False
            flag_crop_type = 0
        global x_offseth
        global y_offseth
        x_offseth = self.y_offseth.text
        y_offseth = self.x_offseth.text
#####################   Makeup Buttons  #####################################################################
    def makeup_options_lips(self, event):
        global flag_makeup
        global flag_makeup_type
        if flag_makeup == False:
            flag_makeup = True
            flag_makeup_type = 1
        else:
            flag_makeup = False
            flag_makeup_type = 0
        global r_input
        global g_input
        global b_input
        global a_input
        a_input = self.a_box.text
        r_input = self.r_box.text
        g_input = self.g_box.text
        b_input = self.b_box.text
        global type_save
        type_save = "Lipstick"

    def makeup_options_eyes(self, event):
        global flag_makeup
        global flag_makeup_type
        if flag_makeup == False:
            flag_makeup = True
            flag_makeup_type = 6
        else:
            flag_makeup = False
            flag_makeup_type = 0
        global r_input
        global g_input
        global b_input
        global a_input
        a_input = self.a_box.text
        r_input = self.r_box.text
        g_input = self.g_box.text
        b_input = self.b_box.text
        global type_save
        type_save = "Eyeliner"

    def makeup_options_brows(self, event):
        global flag_makeup
        global flag_makeup_type
        if flag_makeup == False:
            flag_makeup = True
            flag_makeup_type = 4
        else:
            flag_makeup = False
            flag_makeup_type = 0
        global r_input
        global g_input
        global b_input
        global a_input
        a_input = self.a_box.text
        r_input = self.r_box.text
        g_input = self.g_box.text
        b_input = self.b_box.text
        global type_save
        type_save = "Brows"

    def makeup_options_shadow(self, event):
        global flag_makeup
        global flag_makeup_type
        if flag_makeup == False:
            flag_makeup = True
            flag_makeup_type = 6
        else:
            flag_makeup = False
            flag_makeup_type = 0
        global r_input
        global g_input
        global b_input
        global a_input
        a_input = self.a_box.text
        r_input = self.r_box.text
        g_input = self.g_box.text
        b_input = self.b_box.text
        global type_save
        type_save = "Eyeshadow"

    def makeup_options_delinated(self, event):
        global flag_makeup
        global flag_makeup_type
        if flag_makeup == False:
            flag_makeup = True
            flag_makeup_type = 4
        else:
            flag_makeup = False
            flag_makeup_type = 0
        global r_input
        global g_input
        global b_input
        global a_input
        a_input = self.a_box.text
        r_input = self.r_box.text
        g_input = self.g_box.text
        b_input = self.b_box.text
        global type_save
        type_save = "Delinated Brows"

    def makeup_options_facepaint(self, event):
            global flag_makeup
            global flag_makeup_type
            if flag_makeup == False:
                flag_makeup = True
                flag_makeup_type = 8
            else:
                flag_makeup = False
                flag_makeup_type = 0
            global r_input
            global g_input
            global b_input
            global a_input
            a_input = self.a_box.text
            r_input = self.r_box.text
            g_input = self.g_box.text
            b_input = self.b_box.text
            global type_save
            type_save = "Face Paint"

    def makeup_options_save(self, event):
        global name_save
        global save
        name_save = self.save_box.text
        if name_save == "":
            save = False
        else:
            save = True
            
    def makeup_options_wipe(self, event):
        global wipe
        wipe = True
#####################   Main App        ######################################################################    
    def update(self, dt):
        try:
            # display image from cam in opencv window
            ret, frame = self.capture.read()
            # convert it to texture
            global flag_crop
            global flag_makeup
            if flag_crop == True or flag_makeup:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 0)
                img_copy = np.zeros_like(frame)
                a = []
                points = []
                points_up = []
                for (i, rect) in enumerate(rects):
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    a=list(face_utils.FACIAL_LANDMARKS_IDXS.items()) 
#####################   Main Crop       ######################################################################
                    if flag_crop == True:
                        for (x, y) in shape:
                            #Options for the crop
                            global flag_crop_type
                            global x_offseth
                            global y_offseth                            
                            if flag_crop_type == 1:
                                crop = frame[shape[38, 1]-int(y_offseth): shape[41, 1]+int(y_offseth), shape[37, 0]-int(x_offseth):shape[40, 0]+int(x_offseth)]                                
                            if flag_crop_type == 2:
                                crop = frame[shape[44, 1]-int(y_offseth): shape[47, 1]+int(y_offseth), shape[43, 0]-int(x_offseth):shape[46, 0]+int(x_offseth)]
                            if flag_crop_type == 3:
                                crop = frame[shape[50, 1]-int(y_offseth): shape[57, 1]+int(y_offseth), shape[48, 0]-int(x_offseth):shape[54, 0]+int(x_offseth)]
                            if flag_crop_type == 4:
                                crop = frame[shape[25, 1]-int(y_offseth): shape[25, 1]+int(y_offseth), shape[22, 0]-int(x_offseth):shape[26, 0]+int(x_offseth)]
                            if flag_crop_type == 5:
                                crop = frame[shape[20, 1]-int(y_offseth): shape[20, 1]+int(y_offseth), shape[18, 0]-int(x_offseth):shape[22, 0]+int(x_offseth)]
                            if flag_crop_type == 6:
                                crop = frame[shape[27, 1]-int(y_offseth): shape[21, 1]+int(y_offseth), shape[21, 0]-int(x_offseth):shape[22, 0]+int(x_offseth)]
                            if flag_crop_type == 7:
                                crop = frame[shape[1, 1]-int(y_offseth): shape[4, 1]+int(y_offseth), shape[2, 0]-int(x_offseth):shape[31, 0]+int(x_offseth)]
                            if flag_crop_type == 8:
                                crop = frame[shape[15, 1]-int(y_offseth): shape[12, 1]+int(y_offseth), shape[35, 0]-int(x_offseth):shape[14, 0]+int(x_offseth)]
                            if flag_crop_type == 9:
                                crop = frame[shape[57, 1]-int(y_offseth): shape[8, 1]+int(y_offseth), shape[5, 0]-int(x_offseth):shape[11, 0]+int(x_offseth)]
                        crop_treated = cv2.resize(crop, (frame.shape[1], frame.shape[0]))
                        buf1 = cv2.flip(crop_treated, 0)
#####################   Main Makeup     ######################################################################
                    if flag_makeup == True:
                        global flag_makeup_type
                        global type_save
                        global r_input
                        global g_input
                        global b_input
                        global a_input
                        name, (i, j) = a[flag_makeup_type-1]
                        for (x, y) in shape[i:j]:
                            points.append([x, y])
                            if flag_makeup_type == 6 and type_save == "Eyeliner" or flag_makeup_type == 4 and type_save == "Delinated Brows":
                                points_up.append([x, y-5])
                            elif flag_makeup_type == 6 and type_save == "Eyeshadow":
                                points_up.append([x, y-10])
                            elif flag_makeup_type == 8:
                                points_up.append([x, y-80])
                        if flag_makeup_type == 4: #Brows case
                            points = np.reshape(points, (-1, 1, 2))
                            points_up = np.reshape(points_up, (-1, 1, 2))
                            if type_save == "Brows":
                                cv2.fillPoly(img_copy, [points], (float(r_input),float(g_input),float(b_input)))
                            elif type_save == "Delinated Brows":
                                cv2.fillPoly(img_copy, [points_up], (float(r_input),float(g_input),float(b_input)))
                                cv2.fillPoly(img_copy, [points], (0, 0, 0))
                            name, (i, j) = a[2]
                            points = []
                            points_up = []
                            for (x, y) in shape[i:j]:
                                points.append([x, y])
                                if type_save == "Delinated Brows":
                                    points_up.append([x, y-5])
                            points = np.reshape(points, (-1, 1, 2))
                            points_up = np.reshape(points_up, (-1, 1, 2))
                            if type_save == "Brows":
                                cv2.fillPoly(img_copy, [points], (float(r_input),float(g_input),float(b_input)))
                            elif type_save == "Delinated Brows":
                                cv2.fillPoly(img_copy, [points_up], (float(r_input),float(g_input),float(b_input)))
                                cv2.fillPoly(img_copy, [points], (0, 0, 0))
                        
                        elif flag_makeup_type == 6: #Eyeliner case
                            points = np.reshape(points, (-1, 1, 2))
                            points_up = np.reshape(points_up, (-1, 1, 2))   
                            cv2.fillPoly(img_copy, [points_up], (float(r_input),float(g_input),float(b_input)))
                            cv2.fillPoly(img_copy, [points], (0, 0, 0))
                            
                            points = []
                            points_up = []
                            name, (i, j) = a[4]
                            for (x, y) in shape[i:j]:
                                points.append([x, y])
                                if flag_makeup_type == 6 and type_save == "Eyeliner":
                                    points_up.append([x, y-5])
                                elif flag_makeup_type == 6 and type_save == "Eyeshadow":
                                    points_up.append([x, y-10])
                            points = np.reshape(points, (-1, 1, 2))
                            points_up = np.reshape(points_up, (-1, 1, 2))
                            cv2.fillPoly(img_copy, [points_up], (float(r_input),float(g_input),float(b_input)))
                            cv2.fillPoly(img_copy, [points], (0, 0, 0))
                        
                        elif flag_makeup_type == 8:
                            points = np.reshape(points, (-1, 1, 2))
                            points_up = np.reshape(points_up, (-1, 1, 2))   
                            cv2.fillPoly(img_copy, [points_up], (float(r_input),float(g_input),float(b_input)))
                            cv2.fillPoly(img_copy, [points], (float(r_input),float(g_input),float(b_input)))
                        else:
                            points = np.reshape(points, (-1, 1, 2))
                            cv2.fillPoly(img_copy, [points], (float(r_input),float(g_input),float(b_input)))

                        



                            #cv2.fillPoly(img_copy, [points], (float(r_input),float(g_input),float(b_input)))
                        #integration of the makeup to the normal camarar
                        makeup_treated = cv2.addWeighted(frame, 1.0, img_copy, float(a_input), 0.0)
                        buf1 = cv2.flip(makeup_treated, 0)

                        global save
                        if save == True:
                            global name_save
                            with open("//home//egr//Desktop//cs_ia//makeup_saved.csv", "a") as doc:
                                writer = csv.writer(doc)
                                writer.writerow(["Name: " + str(name_save), " Type: " + type_save, " r: " + r_input, " g: " + g_input, " b: " + b_input, " a: " + a_input])
                                save = False
                        global wipe
                        if wipe == True:
                            with open("//home//egr//Desktop//cs_ia//makeup_saved.csv", "w") as doc:
                                doc.truncate
                                doc.close
                                wipe = False

            else: 
                buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
            #if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer. 
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.img1.texture = texture1
        except:
            pass


if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()