from django.http import HttpResponse
from django.shortcuts import render
from django.shortcuts import render_to_response
from django.core.files.storage import FileSystemStorage
# Create your views here.
#from imageai.Detection import ObjectDetection
#detctor = ObjectDetection()
#detctor.setModelTypeAsRetinaNet()
#detctor.setModelPath("D:\\BE Machine learning\\Final_Headache\\resnet50_coco_best_v2.0.1.h5")
#detctor.loadModel()

def index(request):
    return render(request, "index.html")


def upload(request):

    from keras.models import load_model
    from keras.preprocessing import image
    import numpy as np
    import cv2
    from keras import backend as K
    import xlrd
    from setuptools.command.test import test
    import tensorflow as tf

    if (request.method == 'POST'):
        uploaded_file = request.FILES['doc']
        fs = FileSystemStorage()
        print('Image name is :', uploaded_file.name)
        print('Image size is :', uploaded_file.size)
        fs.save(uploaded_file.name, uploaded_file)
        print("THe model")

        print("THe model is loaded")
        loc = ("D:\\BE Machine learning\\Final_Headache\\scene.xlsx")

        K.set_image_dim_ordering('th')
        # dimensions of our images
        img_width, img_height = 128, 128

        wb = xlrd.open_workbook(loc)
        sheet = wb.sheet_by_index(0)

        #    load the model we saved
        model = load_model('D:\\BE Machine learning\\Final_Headache\\model.hdf5')
        print(model.summary())
        print(model.get_weights())
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

        image_path = ("D:\\BE Machine learning\\Django_new\\mysite\\media\\"+uploaded_file.name)
        # predicting images
        test_image = cv2.imread(image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        test_image = cv2.resize(test_image, (128, 128))
        # r=test_image.reshape(1,128,128,1)
        test_image = np.array(test_image)
        test_image = test_image.astype('float32')
        test_image /= 255
        print(test_image.shape)
        num_channel = 1
        if num_channel == 1:
            if K.image_dim_ordering() == 'th':
                test_image = np.expand_dims(test_image, axis=0)
                test_image = np.expand_dims(test_image, axis=0)
                print(test_image.shape)
            else:
                test_image = np.expand_dims(test_image, axis=3)
                test_image = np.expand_dims(test_image, axis=0)
                print(test_image.shape)

        else:
            if K.image_dim_ordering() == 'th':
                test_image = np.rollaxis(test_image, 2, 0)
                test_image = np.expand_dims(test_image, axis=0)
                print(test_image.shape)
            else:
                test_image = np.expand_dims(test_image, axis=0)
                print(test_image.shape)

        a = model.predict(test_image)
        print((model.predict(test_image)))
        classes = model.predict_classes(test_image)
        print(type(classes))
        a1 = int(classes) + 1
        category = sheet.cell_value(a1, 1)
        per = (a[0][classes] * 100)
        # print(bb)
        print("the category of Scene:", category, " percentage value:", per)
        types = sheet.cell_value(a1, 2)
        print("the types of Scene:" + types)
        attr = sheet.cell_value(a1, 3)
        print("Attributes:" + attr)
        # detections = detctor.detectObjectsFromImage(input_image=image_path,output_image_path='a111.jpg',minimum_percentage_probability=30)

        # for ec in detections:
        # print(ec["name"])
        # print(max(model.predict(test_image[classes])))
        # print(type(model.predict(test_image)))

        total="Category:"+category
        total1="Type:"+types
        total2="attributes:"+attr

        return render_to_response('index.html', {'variable': total})
        #
    else:
        return render(request, 'index.html')