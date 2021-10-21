import streamlit as st
import cv2
import io
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "G_pix2pix_BF2Cy5"
@st.cache(allow_output_mutation=True)
def model_load():
    model = torch.load(model_path).module
    return model

def main():
    st.set_page_config(layout="wide")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    st.markdown("<h1 style='text-align: center; color: white;'>Neural cell in-silico staining using pix2pix conditional GAN</h1>", unsafe_allow_html=True)
    menu = ['Cy5']
    st.sidebar.header('Desired Channel')
    choice = st.sidebar.selectbox('What channel', menu)
    #st.image('banner.png',use_column_width = False
    #)
    

    sample_img = st.file_uploader('Upload your bright field image here',type=['tif','jpg'])
    sample_label = st.file_uploader('Upload your Cy5 image here (if have)',type=['tif','jpg'])
    if sample_img is not None:
        mean_BF = torch.tensor([2355.3508])
        std_BF = torch.tensor([353.2483])
        mean_label = torch.tensor([2185.792])
        std_label = torch.tensor([2148.2190])
        pre_transform = transforms.ToTensor()

        post_transform_image = transforms.Compose([
                                                transforms.Resize(size=(600, 600)),
                                                transforms.CenterCrop(512),
                                                transforms.Normalize(mean_BF,std_BF)
                                                ])
        post_transform_label = transforms.Compose([
                                                transforms.Resize(size=(600, 600)),
                                                transforms.CenterCrop(512),
                                                transforms.Normalize(mean_label,std_label)])
        to_image = transforms.ToPILImage()
        inv_normalize_BF = transforms.Normalize(
            mean=-mean_BF/std_BF,
            std=1/std_BF
        )
        inv_normalize_label = transforms.Normalize(
            mean=-mean_label/std_label,
            std = 1/std_label
        )
        G = model_load().to(device)
        x = pre_transform(Image.open(sample_img)).float()
        x = post_transform_image(x)
        
        if st.button('All files uploaded'):
            with st.spinner("Staining..."):

                if sample_label is not None:
                    y = pre_transform(Image.open(sample_label)).float()
                    y = post_transform_label(y)
                    with torch.no_grad():
                        inv_y = inv_normalize_label(y)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            col1.header("Brightfield Image")
                            fig, ax = plt.subplots(1,1,figsize=(25, 25))
                            plt.rcParams['axes.facecolor'] = 'black'
                            plt.subplot(1,1,1)
                            plt.axis('off')
                            plt.imshow(to_image(10*inv_normalize_BF(x).type(torch.int16).squeeze(0)+10000),cmap = 'cividis')
                            st.pyplot(fig)

                        with col2:
                            col2.header('True '+choice)
                            fig, ax = plt.subplots(1,1,figsize=(25, 25))
                            plt.subplot(1,1,1)
                            plt.axis('off')
                            plt.imshow(to_image(inv_y[0].type(torch.int16).squeeze(0)))
                            st.pyplot(fig)

                        with col3:
                            col3.header('Generated '+choice)
                            fig, ax = plt.subplots(1,1,figsize=(25, 25))
                            plt.rcParams['axes.facecolor'] = 'black'
                            plt.subplot(1,1,1)
                            plt.axis('off')
                            plt.imshow(to_image(inv_normalize_label(G(x.unsqueeze(0).cuda()).squeeze(0))[0].type(torch.int16)))
                            st.pyplot(fig)

                else:
                    with torch.no_grad():
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            col1.header("Brightfield Image")
                            fig, ax = plt.subplots(1,1,figsize=(25, 25))
                            plt.rcParams['axes.facecolor'] = 'black'
                            plt.subplot(1,1,1)
                            plt.axis('off')
                            plt.imshow(to_image(10*inv_normalize_BF(x).type(torch.int16).squeeze(0)+10000),cmap = 'cividis')
                            st.pyplot(fig)
                        
                        with col2:
                            col2.header('Generated '+choice)
                            fig, ax = plt.subplots(1,1,figsize=(25, 25))
                            plt.rcParams['axes.facecolor'] = 'black'
                            plt.subplot(1,1,1)
                            plt.axis('off')
                            plt.imshow(to_image(inv_normalize_label(G(x.unsqueeze(0).cuda()).squeeze(0))[0].type(torch.int16)))
                            st.pyplot(fig)
                        


        
if __name__ == '__main__':
    main()