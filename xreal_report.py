import torch
from PIL import Image
import numpy as np
import pickle
import albumentations
import vqgan
# Assuming your VQGanVAE class and required dependencies are defined in a module named vae
from vqgan.vae import VQGanVAE
from vqgan.unified_plmodel import TransformerLightning_unified
import warnings
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn


kargs_unified = {
    'num_tokens': 30522,
    'num_img_tokens': 1030,
    'img_vocab_size':1024, 
    'max_seq_len': 2308,
    'max_img_len' : 2052,
    'max_img_num': 2,
    'img_len':1026,
    'dim': 768,
    'depth': 12,
    'heads': 12,
    'dim_head': 64,
    'local_attn_heads': 0,
    'local_window_size': 256,
    'causal': True,
    'attn_type': 'all_modality_causal_cuda',
    'nb_features': 64,
    'feature_redraw_interval': 1000,
    'reversible': False,
    'ff_chunks': 1,
    'ff_glu': False,
    'emb_dropout': 0.1,
    'ff_dropout': 0.1,
    'attn_dropout': 0.1,
    'generalized_attention': True,
    'kernel_fn': nn.ReLU(),
    'use_scalenorm': False,
    'use_rezero': False,
    'tie_embed': False,
    'rotary_position_emb': False,
    'img_fmap_size': 32,
    'FAVOR': True,
    'epochs': 200,
    'ckpt_dir': None,
    'under_sample': 'fixed_all_unified',
    'target_count': 1,
    'weights':True
}



class ReportGeneration:
    def __init__(self, device, vqgan_model_path):
        self.indices_dict = {}  # Dictionary to store codebook indices
        self.saved_model_path = None  # Path to save the model
        self.device = device
        self.model_path = vqgan_model_path
    def load_images_from_txt(self, file_path):
        return [file_path]

    # Function to process images, obtain codebook indices, and save the model
    def process_images_and_save_model(self, model_path, config_path, image):
        print('hoooooooooo')
        vqgan_model = VQGanVAE(vqgan_model_path=model_path, vqgan_config_path=config_path)
        print('dslkfjslkgfjsdlgk')
        vqgan_model.to(self.device)
        vqgan_model.eval()

        rescaler = albumentations.SmallestMaxSize(max_size = 512)
        cropper = albumentations.CenterCrop(height=512,width=512)
        preprocessor = albumentations.Compose([rescaler, cropper])

        image = preprocessor(image = image)
        image = (image['image']/127.5 - 1.0).astype(np.float32)

        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            z_q, emb_loss, perplexity, indices = vqgan_model.get_codebook_indices(img_tensor)
        self.indices_dict['1'] = indices.cpu().numpy().tolist()[0]

        return indices
    # Function to load the saved model and decode images
    def decode_and_save_images(self, codebook_indices_path, save_folder):
        vqgan_model = VQGanVAE(vqgan_model_path=self.model_path, vqgan_config_path=self.device)
        vqgan_model.to(self.device)
        vqgan_model.eval()

        # Decode images and save them to the specified folder
        for image_key, indices in self.indices_dict.items():
            indices = torch.tensor(indices).to(self.device)
            decoded_image = vqgan_model.decode(indices)
            decoded_image = decoded_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            decoded_image = (decoded_image * 255).astype(np.uint8)

            image_save_path = f"{save_folder}/{image_key}.png"
            Image.fromarray(decoded_image).save(image_save_path)

