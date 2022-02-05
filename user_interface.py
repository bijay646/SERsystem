import tkinter as tk
import tkinter.messagebox
import pyaudio
import wave
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report

from playsound import playsound

import extraction_functions_rec as efrs
#import main


class RecAUD:

    def __init__(self,chunk=3024 , frmat=pyaudio.paInt16, channels=2, rate=44100, py=pyaudio.PyAudio()):

        # Start Tkinter and set Title
        self.main = tkinter.Tk()
        #self.collections = []
        self.main.geometry('500x300')
        self.main.title('Record')
        self.CHUNK = chunk
        self.FORMAT = frmat
        self.CHANNELS = channels
        self.RATE = rate
        self.p = py
        #self.frames = []
        self.st = 1
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

        # Set Frames
        self.buttons = tkinter.Frame(self.main, padx=130, pady=20)

        # Pack Frame
        self.buttons.pack(fill=tk.BOTH)



        # Start and Stop buttons
        self.strt_rec = tkinter.Button(self.buttons, width=10, padx=10, pady=5, text='Start Recording',bg = "black",fg="white", command=lambda: self.start_record())
        self.strt_rec.grid(row=0, column=0, padx=50, pady=5)
        self.stop_rec = tkinter.Button(self.buttons, width=10, padx=10, pady=5, text='Stop Recording',bg = "black",fg="white", command=lambda: self.stop())
        self.stop_rec.grid(row=1, column=0, columnspan=1, padx=50, pady=5)

        tkinter.mainloop()

    def start_record(self):
        self.st = 1
        self.frames = []
        stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
        while self.st == 1:
            data = stream.read(self.CHUNK)
            self.frames.append(data)
            print("* recording")
            self.main.update()

        stream.close()

        wf = wave.open('test_recording.wav', 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def stop(self):
        self.st = 0


# Create an object of the ProgramGUI class to begin the program.
disp = RecAUD()

#path="C:\\Users\\Dell\\Desktop\\major project\\SERsystem\\test_recording.wav\\"
#playsound('test_recording.wav')



feature = efrs.get_features('test_recording.wav')
X= np.array([])
for ftr in feature:
   X=np.hstack((X,ftr))

print(X)
X = X.reshape(1,-1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

print(X)




