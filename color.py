#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# Coded by Kuduxaaa

import numpy as np
from PIL import Image


შემავალი_შრეები = 3
შიდა_შრეები = 3
გამავალი_შრეები = 3


class ColorRecognition:
    def __init__(self):
        np.random.seed(1)
        self.colors = {
            'black': 0,
            'purple': 0,
            'blue': 0,
            'orange': 0,
            'red': 0,
            'yellow': 0,
            'white': 0,
            'green': 0
        }


    def scale(self, x):
        return x * 1.0 / 255


    def sigmoid(self, x, derivative = False):
        if derivative:
            return (x) * (1 - x)

        return 1 / (1 + np.exp(-x))


    def calculate_colors(self, l1, l2):
        if round(l2[0], 0) == 0 and round(l2[1], 0) == 0 and round(l2[2], 0) == 0:
            self.colors['black'] = (self.colors['black'] + 1)
        if round(l2[0], 0) == 0 and round(l2[1], 0) == 0 and round(l2[2], 0) == 1:
            self.colors['red'] = (self.colors['red'] + 1)
        if round(l2[0], 0) == 0 and round(l2[1], 0) == 1 and round(l2[2], 0) == 0:
            self.colors['blue'] = (self.colors['blue'] + 1)
        if round(l2[0], 0) == 0 and round(l2[1], 0) == 1 and round(l2[2], 0) == 1:
            self.colors['purple'] = (self.colors['purple'] + 1)
        if round(l2[0], 0) == 1 and round(l2[1], 0) == 0 and round(l2[2], 0) == 0:
            self.colors['green'] = (self.colors['green'] + 1)
        if round(l2[0], 0) == 1 and round(l2[1], 0) == 0 and round(l2[2], 0) == 1:
            self.colors['yellow'] = (self.colors['yellow'] + 1)
        if round(l2[0], 0) == 1 and round(l2[1], 0) == 1 and round(l2[2], 0) == 0:
            self.colors['orange'] = (self.colors['orange'] + 1)
        if round(l2[0], 0) == 1 and round(l2[1], 0) == 1 and round(l2[2], 0) == 1:
            self.colors['white'] = (self.colors['white'] + 1)



    def recognizeColorFromImage(self, image):
        '''
        სურათის ანალიზი, გახსნა და პიქსელებათ დაშლა
        '''
        opened_image = Image.open(image)
        pixels = opened_image.load()
        w, h = opened_image.size
        image_total_size = (w * h)
        trainer = Train()
        syn0, syn1 = trainer.train_rgb()

        for eh in range(0,h):
            for ew in range(w):
                R,G,B = opened_image.getpixel((ew, eh))
                scales = [self.scale(R), self.scale(G), self.scale(B)]

                l0 = scales
                l1 = self.sigmoid(np.dot(l0, syn0))
                l2 = self.sigmoid(np.dot(l1, syn1))
                self.calculate_colors(l1, l2)

        for color in self.colors:
            print(color + ": " + str(int((float(self.colors[color]) / image_total_size) * 100)) + "%")



class Train(ColorRecognition):
    def __init__(self):
        super().__init__()
        np.random.seed(1)

    def train_rgb(self):
        ''' ნეირონები '''

        X = np.array([
            # ნარინჯისფერი
            [self.scale(250), self.scale(181), self.scale(127)],
            [1, self.scale(165), 0],
            [1, self.scale(165), 0],

            # ლურჯი
            [self.scale(207), self.scale(75), self.scale(65)],
            [self.scale(173), self.scale(216), self.scale(230)],
            [0,0, self.scale(139)],

            # მწვანე
            [self.scale(144), self.scale(238), self.scale(144)],
            [0,self.scale(100), 0],

            # ყვითელი 
            [self.scale(239), self.scale(239), self.scale(74)],
            [self.scale(242), self.scale(242), self.scale(62)],

            # ნაცრისფერი 
            [self.scale(211), self.scale(211), self.scale(211)],
            [self.scale(169), self.scale(169), self.scale(169)],

            # შავი
            [self.scale(4), self.scale(11), self.scale(19)],
            [self.scale(0), self.scale(0), self.scale(7)],
            [self.scale(8), self.scale(11), self.scale(28)],

            # თეთრი
            [self.scale(255), self.scale(255), self.scale(255)],
            [self.scale(254), self.scale(252), self.scale(250)]
        ])

        Y = [
            [1,1,0], [1,1,0],
            [0,0,1], [0,0,1],
            [0,1,0], [0,1,0],
            [1,0,0], [1,0,0],
            [1,0,1], [1,0,1],
            [0,1,1], [0,1,1],
            [0,0,0], [0,0,0], 
            [0,0,0], [1,1,1],
            [1,1,1]
        ]

        '''
        სინაფსები
        '''
        syn0 = (2 * np.random.random((შემავალი_შრეები, შიდა_შრეები)) - 1)
        syn1 = (2 * np.random.random((შიდა_შრეები, გამავალი_შრეები)) - 1)


        '''
        10,000 ნეირონისა და სინაფსის ურთიერთქმედება
        '''
        for i in range(10000):
            l0 = X
            l1 = self.sigmoid((np.dot(l0, syn0)))
            l2 = self.sigmoid((np.dot(l1, syn1)))

            l2_error = (Y - l2)
            l2_delta = (l2_error * self.sigmoid(l2, derivative = True))

            l1_error = l2_error.dot(syn1.T)
            l1_delta = (l1_error * self.sigmoid(l1, derivative = True))

            syn1 += np.dot(l1.T, l2_delta)
            syn0 += np.dot(l0.T, l1_delta)

        return (syn0, syn1)



if __name__ == '__main__':
    neural_network = ColorRecognition()
    neural_network.recognizeColorFromImage('plant.jpeg')