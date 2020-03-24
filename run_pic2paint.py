# -*- coding: utf-8 -*-
"""
Streamlit app for Pic2Paint

Author: Michael Riedl
Last Updated: March 18, 2020
"""

import torch
import scipy.linalg
import numpy as np
import streamlit as st
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from utils.models import StyleTransferNet

def load_model(device, content_img, style_img):
    loader = transforms.Compose([transforms.ToTensor()])
    return StyleTransferNet(device, loader(style_img).unsqueeze(0), loader(content_img).unsqueeze(0))

def color_balance(content_img, style_img):
    # Store the size of the original style image
    imsize = style_img.size
    # Convert the content and style images to tensors
    loader = transforms.Compose([transforms.ToTensor()])
    content_img = loader(content_img)
    style_img = loader(style_img)
    # Create the style covariance
    style_vec = np.moveaxis(style_img.cpu().numpy(), 0 , -1).reshape(-1, 3)
    style_mean = np.mean(style_vec, axis=0)
    style_cov = 1/style_vec.shape[0]*np.matmul(np.transpose(style_vec - style_mean), style_vec - style_mean)
    # Create the content covariance
    content_vec = np.moveaxis(content_img.cpu().numpy(), 0 , -1).reshape(-1, 3)
    content_mean = np.mean(content_vec, axis=0)
    content_cov = 1/content_vec.shape[0]*np.matmul(np.transpose(content_vec - content_mean), content_vec - content_mean)
    # Apply the color balance to the style image
    style_img_conv = np.reshape(style_img.cpu().numpy(), (3, -1))
    A = np.matmul(scipy.linalg.sqrtm(content_cov), scipy.linalg.sqrtm(np.linalg.inv(style_cov)))
    b = content_mean - np.matmul(A, style_mean)
    style_img_conv = np.clip(np.matmul(A, style_img_conv) + b[:, None], 0, 1)
    style_img_conv = torch.tensor(np.reshape(style_img_conv, (3, imsize[1], imsize[0])))
    saver = transforms.Compose([transforms.ToPILImage()])
    style_img_conv = saver(style_img_conv)
    
    return style_img_conv
    
# Remove old variables
model = None

# Create the headers
st.title("Pic2Paint")
st.header("Upload images to start.")

# Initialize the sidebar view
st.sidebar.header("Processing Status")
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
status_text.text("Waiting to process...")

# Create the file uploaders
st.sidebar.header("Choose the pictures to use")
content_img_file = st.sidebar.file_uploader("Content Image")
style_img_file = st.sidebar.file_uploader("Style Image")

# Create the resize options
resize_flag = st.sidebar.selectbox("Resizing", ["None", "Content", "Style"])
color_flag = st.sidebar.checkbox("Maintain content color")

# Display the chosen images and enable run if valid
if(content_img_file is not None and style_img_file is not None):
    content_img = Image.open(content_img_file)
    style_img = Image.open(style_img_file)
    # Perform resizing
    if(resize_flag == 'Content'):
        style_img = ImageOps.fit(style_img, content_img.size, Image.ANTIALIAS)
    if(resize_flag == 'Style'):
        content_img = ImageOps.fit(content_img, style_img.size, Image.ANTIALIAS)
    # Perform color balance
    if(color_flag):
        style_img = color_balance(content_img, style_img)
    st.image([content_img, style_img], 
             caption=['Content Image', 'Style Image'], 
             width=256)
    st.sidebar.header("Run style transfer")
    proc_flag = st.sidebar.button("Run")
elif(content_img_file is not None):
    content_img = Image.open(content_img_file)
    st.image([content_img], 
             caption=['Content Image'], 
             width=256)
elif(style_img_file is not None):
    style_img = Image.open(style_img_file)
    st.image([style_img], 
             caption=['Style Image'], 
             width=256)
   
# Run the style transfer
if(content_img_file is not None and style_img_file is not None and proc_flag):
    
    # Set the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = load_model(device, content_img, style_img)
    
    # Change the status text
    status_text.text("Processing...")
    # Set the number of processing steps
    max_steps = 60
    
    # Initialize list for creating gif
    final_gif = []
    saver = transforms.Compose([transforms.ToPILImage()])

    # Create the input
    loader = transforms.Compose([transforms.ToTensor()])
    input_img = loader(content_img).unsqueeze(0).to(device)
    optimizer = optim.LBFGS([input_img.requires_grad_()], max_iter=1, history_size=1)
    step = [0]
    while step[0] < max_steps:
        
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            model.loss.backward()
            step[0] += 1
            return model.loss.item()
        
        optimizer.step(closure)
        progress_bar.progress(step[0]/max_steps)
        # Store the intermediate image for the gif
        input_img.data.clamp_(0, 1)
        final_gif.append(saver(input_img.cpu().squeeze(0)))
        
    # Change the status text
    status_text.text("Complete!")
    
    # Plot the final image
    final_img = saver(input_img.cpu().squeeze(0))
    st.image([final_img], 
             caption=['Stylized Image'], 
             width=256)
    
    # Save the image
    final_img.save("output.png")
    # Save the gif
    final_gif[0].save("output.gif", save_all=True, append_images=final_gif[1:])
        