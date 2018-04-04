'''
*** Dajngo Core packages ***
'''

from django.shortcuts import render, HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import os, uuid

'''
*** Packages for Image Processing ***
'''
import numpy as np
import pandas as pd
import cv2
from PIL import Image

'''
*** Packages for OCR ***
'''
import pytesseract

'''
*** Packages for text to speech ***
'''
from gtts import gTTS

'''
*** Packages for Barcode Data Read ***
'''
import requests
import json
from pyzbar.pyzbar import decode

'''
*** Color Detection Modules ***
'''
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import utils
from sklearn import metrics
import webcolors



def centroid_histogram(clt):
    '''
    Various "color" cluters are imported and creates a histogram based 
    on the number of pixels assigned to each cluster.
    '''
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
    '''
    The histogram is normalized to lie within the range of [0,1]. Based 
    on the centroid of each cluster, the color can be detected.
    '''
    hist = hist.astype("float")
    hist /= hist.sum()

    '''
    The normalized histogram is returned for further processing
    '''
    return hist

def plot_colors(hist, centroids):
    '''
    Initialization of barchart which represents the relative frequency
    of each of the colors present in the image
    '''
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
    
    '''
    Loops over the percentage of each cluster and the color of each cluster.
    '''
    for (percent, color) in zip(hist, centroids):
        '''
        Plots the relative percentage of each cluster for "color" detection
        '''
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX

    '''
    Returns barchart for further processing
    '''
    return bar

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        color_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        color_name = closest_colour(requested_colour)
    return color_name


def generate_audio(text):
    tts = gTTS(text = text, lang = 'en')
    uuid_name = uuid.uuid4()
    new_filename = '/' + str(uuid_name) + '.mp3'
    tts.save(settings.MEDIA_ROOT + new_filename)
    return new_filename


def detect_text(file_name):
    image = cv2.imread(settings.MEDIA_ROOT + '/' +  file_name)
    text = pytesseract.image_to_string(image)
    new_filename = generate_audio(text)
    return new_filename


def detect_color(file_name):    
    image = cv2.imread(settings.MEDIA_ROOT + '/' +  file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# show our image
	#plt.figure()
	#plt.axis("off")
	#plt.imshow(image)

	# reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))


    # cluster the pixel intensities
    clt = KMeans(n_clusters = 8, n_jobs = -1)
    clt.fit(image)

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)

    # show our color bart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)
    colors = []
    for item in clt.cluster_centers_:
        colors += [(int(item[0]), int(item[1]), int(item[2]))]

    readout_text = ["The Dominant colors in the image are:"]
    for item in colors:
        requested_colour = item
        color_name = get_colour_name(requested_colour)
        readout_text.extend((color_name, ','))


    print(readout_text) #values of colors
    text = " ".join(readout_text)
    print (text)
    new_filename = generate_audio(text)
    return new_filename


def detect_barcode(file_name):
    #Load Image
    image = cv2.imread(settings.MEDIA_ROOT + '/' +  file_name)
    image_barcode = cv2.imread("/home/bhavishk/Project_Disk/Final Year Project/Image Processing/Images/Src/barcode1.jpg")
    #image_barcode = cv2.imread(settings.MEDIA_ROOT + '/' +  file_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.blur(gradient, (9,9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    (_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    # draw a bounding box arounded the detected barcode and display the
    # image
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    imS = cv2.resize(image, (960, 540))
    cv2.imshow("output", imS)   
    cv2.waitKey(0)

    #Detect Barcode from the Image
    barcode = decode(image_barcode)
    print (barcode)
    barcode = str(barcode[0][0])
    print (barcode)
    barcode = barcode[2:-1]
    print (barcode)
    #Retrival of data based on barcode from the upcitemdb
    url = "https://api.upcitemdb.com/prod/trial/lookup?upc=" + barcode
    r = requests.get(url)
    json_data = r.text

    #extracting data from json
    parsed = json.loads(json_data)
    #ean = parsed["items"][0]["ean"]
    title = parsed["items"][0]["title"]
    brand = parsed["items"][0]["brand"]
    description = parsed["items"][0]["description"]


    length = len(parsed["items"][0]["offers"])
    index = [x for x in range(0,length)]
    df = pd.DataFrame(index = index, columns = {'merchant', 'price'})
    for counter, item in enumerate(parsed["items"][0]["offers"]):
        if parsed["items"][0]["offers"][counter]["price"] > 0:
            df.iloc[counter][0] = parsed["items"][0]["offers"][counter]["merchant"]
            df.iloc[counter][1] = parsed["items"][0]["offers"][counter]["price"]

    df = df.dropna()
    #Nutrition Data
    print (df)

    url = 'https://trackapi.nutritionix.com/v2/natural/nutrients'
    headers = {
        'Accept': 'application/json', 
        'x-app-id': '4d809d7e', 
        'x-app-key': '3725482902de1514e88104b39d12b6ca', 
        'x-remote-user-id': '0'
    }

    data = json.dumps({'query': brand})
    response = requests.post(
        'https://trackapi.nutritionix.com/v2/natural/nutrients', 
        headers = headers, 
        data = data
    )
    nutrition_dump = json.loads(response.text)
    calories = nutrition_dump["foods"][0]["nf_calories"]
    total_fat = nutrition_dump["foods"][0]["nf_total_fat"]
    saturated_fat = nutrition_dump["foods"][0]["nf_saturated_fat"]
    cholesterol = nutrition_dump["foods"][0]["nf_cholesterol"]
    sodium = nutrition_dump["foods"][0]["nf_sodium"]
    carbohydrate = nutrition_dump["foods"][0]["nf_total_carbohydrate"]
    sugars = nutrition_dump["foods"][0]["nf_sugars"]
    protein = nutrition_dump["foods"][0]["nf_protein"]
    potassium = nutrition_dump["foods"][0]["nf_potassium"]


    #Readout Text Generation

    readout_text = ['The Requested Product']
    readout_text.extend((
        str(title),
        ', is a, ',
        str(description),
        '.',
        ', The requested product has the following nutritional values, ',
        ', calories, ', 
        str(calories), 
        ', Kcal, ',
        ', total fat, ',
        str(total_fat),
        ', grams, saturated fat, ',
        str(saturated_fat),
        ', grams,  cholesterol, ',
        str(cholesterol),
        ', grams, sodium, ',
        str(sodium),
        ', grams, carbohydrates',
        str(carbohydrate),
        ', grams, sugars',
        str(sugars),
        ', grams, protein',
        str(protein),
        ', grams, potassium',
        str(potassium),
        ', grams, .',
        'From our search on multiple platforms, these are the selling price at various retailers.',
    ))
    for index in range(len(df)):
        readout_text.extend((
            ',',
            str(df.ix[index][0]), 
            ', sells the products at,', 
            str(df.ix[index][1]),
            ', dollars,',
        ))
    text = " ".join(readout_text)
    new_filename = generate_audio(text)
    return new_filename

@csrf_exempt
def upload(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        mode = request.POST['mode']
        #print(mode)
        fs = FileSystemStorage()
        file_name = fs.save(image.name, image)
        #print (file_name)
        #print (settings.MEDIA_URL)

        if mode == "barcode_detect":
            '''
            f = open(settings.MEDIA_ROOT + '/text.txt')
            for item in f:
                print (item)
            '''
            audio_filename = detect_barcode(file_name)
            
        if mode == 'text_detect':
            audio_filename = detect_text(file_name)
            
        if mode == 'color_detect':
            audio_filename = detect_color(file_name)

        fname = settings.MEDIA_ROOT + audio_filename
        f = open(fname,"rb") 
        response = HttpResponse()
        response.write(f.read())
        response['Content-Type'] ='audio/mp3'
        response['Content-Length'] =os.path.getsize(fname)
        return response
    return render(request, 'upload.html')
        
