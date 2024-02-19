#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
import sys
import requests
sys.path.append('/home/aurelien/miniconda3/lib/python3.8/site-packages')
from io import BytesIO
from PIL import Image, ImageDraw
import random
import os
import time
import numpy as np
import skimage
from rembg import remove
import pickle
from skimage import io, color, filters, measure, morphology

#https://huggingface.co/microsoft/git-large-r-coco
# "https://api-inference.huggingface.co/models/michelecafagna26/git-base-captioning-ft-hl-scenes"
API_URL_img2text = "https://api-inference.huggingface.co/models/microsoft/git-large-r-coco"
headers_img2text = {"Authorization": f"Bearer {'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'}"}

def query_img2text(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL_img2text, headers=headers_img2text, data=data, timeout=60)
    if response.status_code == 200:
        return response.json()
    else :
        print("failed request img2text:", response.status_code)
#output[0]['generated_text'] texte descripteur
#print(type(output),output[0]['generated_text'])

#https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/discussions/127
API_URL_text2img = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers_text2img = {"Authorization": "Bearer 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"}

def query_text2img(payload):
	response = requests.post(API_URL_text2img, headers=headers_text2img, json=payload, timeout=60)
	return response.content

# You can access the image with PIL.Image for example
#image = Image.open(io.BytesIO(image_bytes))


def square_select(image_name):
    image = Image.open(image_name)

    size = random.randint(250, 723)
    max_coordinate = 1024 - size #moins de la moitié de l'image
    x = random.randint(0,max_coordinate)
    y = random.randint(0,max_coordinate)

    square_region = image.crop((x, y, x + size, y + size))
    square_region.save('squares/'+image_name[53:-4]+"_auth_square.png") #0:-4 => image.png

    return (x,y,size)

def build_image(image_name, alpha_threshold = 127):
    x,y,size = square_select(image_name)
    start = time.time()
    #create forged_square
    while True:
        try:
            text = query_img2text('squares/'+image_name[53:-4]+"_auth_square.png")[0]['generated_text']
            time.sleep(4)
            print('temps de caclul réalisé = ', time.time()-start , 's')
            break
        except TypeError as e:
            print('Error : ', e)
            time.sleep(4)
    time_img2text = time.time()-start
    print('img2text time = ',time_img2text)
    
    forged_bytes = query_text2img({"inputs" : text,})
    forged_stream = BytesIO(forged_bytes)
    forged_square = Image.open(forged_stream)
    forged_square = forged_square.resize((size,size))
    
    #remove background
    forged_array = np.array(forged_square)
    rmbg_array = np.copy(remove(forged_array))
    rmbg_array[:,:,3] = np.where(rmbg_array[:,:,3] >= alpha_threshold,255,0)
    forged_square = Image.fromarray(rmbg_array, 'RGBA')
    forged_square.save('squares/'+image_name[53:-4]+'_forged_square.png')
    
    #paste forged_square
    auth_img = Image.open(image_name)
    forged_img = Image.new("RGB",auth_img.size,"white")
    forged_img.paste(auth_img,(0,0))
    forged_img.paste(forged_square, (x,y), forged_square)
    forged_img.save('forged/'+image_name[53:-4]+'_forged.png')
    #save forgery mask
    mask = Image.new("RGB", (1024, 1024), "white")
    black = Image.new("RGB", (size,size),"black")
    mask.paste(black, (x,y), forged_square)
    mask.save('masks/'+image_name[53:-4]+'_mask.png')
    end = time.time()
    print('execution time =', end - start)
    print(text)

    prompts = "prompts.pikl"
    try:
        with open(prompts, "rb") as f:
            dic = pickle.load(f)
    except FileNotFoundError:
        dic = {}
    dic[image_name] = text
    with open(prompts,"wb") as f:
        pickle.dump(dic,f)
    return text



def build_data(auth_path):
    prompts = "prompts.pikl"
    try:
        with open(prompts, "rb") as f:
            dic = pickle.load(f)
    except FileNotFoundError:
        dic = {}
    start = time.time()
    image_list = os.listdir(auth_path)
    for image in image_list:
        if not image in dic:
            print(image)       
            #text = build_image(auth_path +'/'+ image)
            #dic[image] = text
    with open(prompts,"wb") as f:
        pickle.dump(dic,f)
    end = time.time()
    print('dataset building time = ', end - start)



#build_data('/home/aurelien/Bureau/build_forged_dataset/authentic')
