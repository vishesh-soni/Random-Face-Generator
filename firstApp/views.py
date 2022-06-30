from django.shortcuts import render
import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model

# Create your views here.
from django.core.files.storage import FileSystemStorage


def home(request):
    return render(request,'home.html')

def predict(request):
    generator=load_model('./models/generator.h5')
    noise = np.random.normal(0, 1, (1, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = np.reshape(gen_imgs,(28,28,3))
    a = image.array_to_img(gen_imgs)
    no = np.random.normal(0,1, (1, 1))
    no = no[0][0]
    a.save('./media'+"/"+str(no)+".jpg")
    img = './media'+"/"+str(no)+".jpg"
    context = {'img':img}
    return render(request,'predict.html',context)


