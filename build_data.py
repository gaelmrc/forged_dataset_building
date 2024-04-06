import sys
import requests
from io import BytesIO
from PIL import Image
import random
import os
import time
import numpy as np
import skimage
from skimage import io, color, filters, measure, morphology
from rembg import remove
import pickle


your_API_key = ''

API_URL_img2text = "https://api-inference.huggingface.co/models/microsoft/git-large-r-coco"
headers_img2text = {"Authorization": f"Bearer {your_API_key}"}

def query_img2text(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL_img2text, headers=headers_img2text, data=data, timeout=60)
    if response.status_code == 200:
        return response.json()
    else :
        print("failed request img2text:", response.status_code)
#output[0]['generated_text'] => text captioned

API_URL_text2img = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers_text2img = {"Authorization": f"Bearer {your_API_key}"}

def query_text2img(payload):
	response = requests.post(API_URL_text2img, headers=headers_text2img, json=payload, timeout=60)
	return response.content

def square_select(image_path, authentic_square_path):
    image = Image.open(image_path)
    width, height = image.size
    size = random.randint(int(0.25*np.sqrt(width*height)), int(np.sqrt(0.5*width*height)))
    max_coordinate = np.min((width,height)) - size 
    x = random.randint(0,max_coordinate)
    y = random.randint(0,max_coordinate)

    square_region = image.crop((x, y, x + size, y + size))
    square_region.save(authentic_square_path) 
    
    return (x,y,size)

def build_image(image_name, authentic_square_path, forged_square_path, forgery_mask_path, forged_image_path, alpha_threshold = 127):
    x,y,size = square_select(image_name)
    #create forged_square
    while True:
        try:
            text = query_img2text(authentic_square_path)[0]['generated_text']
            time.sleep(4)
            break
        except TypeError as e:
            print('Error : ', e)
            time.sleep(4)
        
    forged_bytes = query_text2img({"inputs" : text,})
    forged_stream = BytesIO(forged_bytes)
    forged_square = Image.open(forged_stream)
    forged_square = forged_square.resize((size,size))
    
    #remove background
    forged_array = np.array(forged_square)
    rmbg_array = np.copy(remove(forged_array))
    rmbg_array[:,:,3] = np.where(rmbg_array[:,:,3] >= alpha_threshold,255,0)
    forged_square = Image.fromarray(rmbg_array, 'RGBA')
    forged_square.save(forged_square_path)
    
    #paste forged_square
    auth_img = Image.open(image_name)
    width, height = auth_img.size
    forged_img = Image.new("RGB",auth_img.size,"white")
    forged_img.paste(auth_img,(0,0))
    forged_img.paste(forged_square, (x,y), forged_square)
    forged_img.save(forged_image_path)
    #save forgery mask
    mask = Image.new("RGB", (width, height), "white")
    black = Image.new("RGB", (size,size),"black")
    mask.paste(black, (x,y), forged_square)
    mask.save(forgery_mask_path)

    return text



def build_data(data_pth, auth_sq_pth, frgd_sq_pth, frgry_msk_pth, frgd_img_pth):
    prompts = "prompts.pikl"
    try:
        with open(prompts, "rb") as f:
            dic = pickle.load(f)
    except FileNotFoundError:
        dic = {}
    image_list = os.listdir(data_pth)
    for image in image_list:
        if not image in dic:
            path1,path2,path3,path4 = image + data_pth, image + frgd_sq_pth, image + frgry_msk_pth, image + frgd_img_pth
            text = build_image(image_name, path1 , path2, path3, path4)
            print (f'{image} => {text}')
            dic[image] = text
    with open(prompts,"wb") as f:
        pickle.dump(dic,f)


#Run build_data on your authentic image dataset to build the forged dataset.
