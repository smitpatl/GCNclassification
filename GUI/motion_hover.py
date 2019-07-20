import tkinter as tk
from tkinter import *
import cv2
import PIL.Image, PIL.ImageTk
from collections import defaultdict



class DynamicHover(tk.Canvas):
    """Trying to dynamically bind a hover object"""
    
    def __init__(self):
        tk.Canvas.__init__(self)
        self.index=0
        self.lines=[]
        self.ovals=[]
        self.onOval = False
        self.branching = False
        self.lastBranch=0
        self.edgedict = defaultdict(list)
        self.points=[]
        self.labels=[]
        
    #def on_enter_line(self, event):
        #canvas.itemconfig(CURRENT,fill='green')

    #def on_leave_line(self, enter):
        #canvas.itemconfig(CURRENT,fill='white')

    def on_enter_oval(self, enter):
        #canvas.itemconfig(CURRENT,outline='green')
        self.onOval = True

    def on_leave_oval(self, event):
        #canvas.itemconfig(CURRENT,outline='yellow')
        self.onOval = False

    def on_click_oval(self, event):
        canvas.itemconfig(CURRENT,fill='red', outline='red')
        now=canvas.coords(CURRENT)
        x, y = event.x, event.y
        print(x,y)
        canvas.old_coords = x, y
        self.branching = True
        
        for i in range(len(self.points)):
            if self.points[i]==[now[0],now[1]]:
                self.lastBranch=i
                break
            


    def art_on_click(self, event):
        if not self.onOval:
            x, y = event.x, event.y
            if canvas.old_coords:
                self.index+=1
                file = open('Vertex-Edge Info.txt', "a+")
                x1, y1 = canvas.old_coords
                self.lines.append(canvas.create_line(x, y, x1, y1, fill='white', width=2, activefill='green'))
                last_click=(x,y)
                #canvas.tag_bind(self.lines[self.index],'<Enter>',self.on_enter_line)
                #canvas.tag_bind(self.lines[self.index],'<Leave>',self.on_leave_line)
                if(not self.branching):
                    self.ovals.append(canvas.create_oval(x1,y1,x1+4,y1+4, fill='yellow', activefill='green'))
                    canvas.tag_bind(self.ovals[len(self.ovals)-1],'<Enter>',self.on_enter_oval)
                    canvas.tag_bind(self.ovals[len(self.ovals)-1],'<Leave>',self.on_leave_oval)
                    canvas.tag_bind(self.ovals[len(self.ovals)-1],'<Button-1>',self.on_click_oval)
                    canvas.create_oval(x,y,x+4,y+4, fill='yellow', activefill='green')
                    #file.write("\n" + "V:" + str(last_click) +"-->")
                    #file.write("  " + str(last_click))
                    
                    #canvas.create_oval(x,y,x+4,y+4, fill='yellow', activefill='green')
                    self.edgedict[self.index-1].append(self.index)
                    self.edgedict[self.index].append(self.index-1)
                    
                else:
                    canvas.create_oval(x,y,x+4,y+4, fill='yellow', activefill='green')
                    #canvas.itemconfig(self.ovals[self.lastBranch],fill='red', outline='red')
                    self.edgedict[self.lastBranch].append(self.index)
                    self.edgedict[self.index].append(self.lastBranch)
                    self.branching=False

                file.close()
            self.ovals.append(canvas.create_oval(x,y,x+4,y+4, fill='yellow', activefill='green'))
            canvas.tag_bind(self.ovals[len(self.ovals)-1],'<Enter>',self.on_enter_oval)
            canvas.tag_bind(self.ovals[len(self.ovals)-1],'<Leave>',self.on_leave_oval)
            canvas.tag_bind(self.ovals[len(self.ovals)-1],'<Button-1>',self.on_click_oval)
            canvas.old_coords = x, y
            print(x,y)
            self.points.append([x,y])
            self.labels.append(0)
            

    def art_vertex(self,event):
        x1, y1 = event.x, event.y
        canvas.old_coords = x1,y1
        self.points.append([x1,y1])
        print(x1,y1)
        self.labels.append(0)
        canvas.create_oval(x1,y1,x1+4,y1+4, fill='white')
        last_click = (x1,y1)
        file = open('Vertex-Edge Info.txt', "a+")
        file.write("\n" + "V:" + str(last_click) +"-->")
        file.close()
        self.index+=1

    def vein_on_click(self, event):
        if not self.onOval:
            x, y = event.x, event.y
            if canvas.old_coords:
                self.index+=1
                file = open('Vertex-Edge Info.txt', "a+")
                x1, y1 = canvas.old_coords
                self.lines.append(canvas.create_line(x, y, x1, y1, fill='white', width=2, activefill='green'))
                last_click=(x,y)
                #canvas.tag_bind(self.lines[self.index],'<Enter>',self.on_enter_line)
                #canvas.tag_bind(self.lines[self.index],'<Leave>',self.on_leave_line)
                if(not self.branching):
                    self.ovals.append(canvas.create_oval(x1,y1,x1+4,y1+4, fill='yellow', activefill='green'))
                    canvas.tag_bind(self.ovals[len(self.ovals)-1],'<Enter>',self.on_enter_oval)
                    canvas.tag_bind(self.ovals[len(self.ovals)-1],'<Leave>',self.on_leave_oval)
                    canvas.tag_bind(self.ovals[len(self.ovals)-1],'<Button-1>',self.on_click_oval)
                    canvas.create_oval(x,y,x+4,y+4, fill='yellow', activefill='green')
                    #file.write("\n" + "V:" + str(last_click) +"-->")
                    #file.write("  " + str(last_click))
                    
                    #canvas.create_oval(x,y,x+4,y+4, fill='yellow', activefill='green')
                    self.edgedict[self.index-1].append(self.index)
                    self.edgedict[self.index].append(self.index-1)
                    
                else:
                    canvas.create_oval(x,y,x+4,y+4, fill='yellow', activefill='green')
                    #canvas.itemconfig(self.ovals[self.lastBranch],fill='red', outline='red')
                    self.edgedict[self.lastBranch].append(self.index)
                    self.edgedict[self.index].append(self.lastBranch)
                    self.branching=False

                file.close()
            self.ovals.append(canvas.create_oval(x,y,x+4,y+4, fill='yellow', activefill='green'))
            canvas.tag_bind(self.ovals[len(self.ovals)-1],'<Enter>',self.on_enter_oval)
            canvas.tag_bind(self.ovals[len(self.ovals)-1],'<Leave>',self.on_leave_oval)
            canvas.tag_bind(self.ovals[len(self.ovals)-1],'<Button-1>',self.on_click_oval)
            canvas.old_coords = x, y
            print(x,y)
            self.points.append([x,y])
            self.labels.append(1)
            

    def vein_vertex(self,event):
        x1, y1 = event.x, event.y
        canvas.old_coords = x1,y1
        self.points.append([x1,y1])
        print(x1,y1)
        self.labels.append(1)
        canvas.create_oval(x1,y1,x1+4,y1+4, fill='white')
        last_click = (x1,y1)
        self.index+=1

    def save(self,event):
        print(self.edgedict,len(self.labels),self.labels)
        file = open('GroundTruth.edges', "a+")
        for i in range(len(self.edgedict)):
            for j in self.edgedict[i]:
                file.write(str(i)+" "+str(j)+"\n")
        file.close()
        file = open('GroundTruth.labels', "a+")
        for i in range(len(self.labels)):
            file.write(str(i)+ " "+str(self.labels[i])+"\n")
        file.close()
        

root = tk.Tk()

topFrame = Frame(root)
topFrame.pack()
imgFrame = Frame(root)
imgFrame.pack(side=BOTTOM)

button1 = Button(topFrame, text = "Save" , fg="purple")
button2 = Button(topFrame, text = "Load Obj file" , fg="purple")
button1.pack(side = "left")
button2.pack(side = "right")

def openImg():
	print("give path here")

def loadObj():
	print("Load existing obj and show existing V, E")


 # Load an image using OpenCV
cv_img = cv2.cvtColor(cv2.imread("01_test.tif"), cv2.COLOR_BGR2RGB)
# Get the image dimensions (OpenCV stores image data as NumPy ndarray)
height, width, no_channels = cv_img.shape
# Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))


canvas = tk.Canvas(root, width=width, height=height)
canvas.create_image(0, 0, image=photo, anchor=NW)
canvas.pack(side = "bottom")
canvas.old_coords = None

obj = DynamicHover();

canvas.bind('<Button-1>', obj.art_on_click)
canvas.bind('<Control-1>', obj.art_vertex)
button1.bind('<Button-1>', obj.save)
canvas.bind('<Button-3>', obj.vein_on_click)
canvas.bind('<Control-3>', obj.vein_vertex)
root.mainloop()
