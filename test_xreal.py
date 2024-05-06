import torch
from PIL import Image
import numpy as np
import pickle
import albumentations

# Assuming your VQGanVAE class and required dependencies are defined in a module named vae
from vae import VQGanVAE
from unified_plmodel import TransformerLightning_unified
import warnings
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn


cache_direc = "./biomed_VLP/"
# Load the model and tokenizer
url = "microsoft/BiomedVLP-CXR-BERT-specialized"
tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True, cache_dir = cache_direc)
tokenizer.add_special_tokens({"additional_special_tokens":["[PAD]", "[CLS]", "[SEP]", "[MASK]"]}) #sansan


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


model = TransformerLightning_unified(
    lr=1e-6,
    weight_decay=1e-6,
    tokenizer=tokenizer,
    pad_token_idx=tokenizer.convert_tokens_to_ids("[PAD]"),
    sos_token_idx=tokenizer.convert_tokens_to_ids("[CLS]"),
    eos_token_idx=tokenizer.convert_tokens_to_ids("[SEP]"),
    # save_dir='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/exp-5-CXRBERT_cls_token',
    save_dir='',

    causal_trans='conditioned_causal',
    **kargs_unified,
)

class YourTestClass:
    def __init__(self):
        self.indices_dict = {}  # Dictionary to store codebook indices
        self.saved_model_path = None  # Path to save the model

    def load_images_from_txt(self, file_path):
        return [file_path]

    # Function to process images, obtain codebook indices, and save the model
    def process_images_and_save_model(self, model_path, config_path, image_paths_file):
        file_paths = self.load_images_from_txt(image_paths_file)

        vqgan_model = VQGanVAE(vqgan_model_path=model_path, vqgan_config_path=config_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vqgan_model.to(device)
        vqgan_model.eval()

        for i, file_path in enumerate(file_paths):
            try:
                image = Image.open(file_path)#.convert('RGB')
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                image = np.array(image).astype(np.uint8)
                rescaler = albumentations.SmallestMaxSize(max_size = 512)
                cropper = albumentations.CenterCrop(height=512,width=512)
                preprocessor = albumentations.Compose([rescaler, cropper])

                image = preprocessor(image = image)
                image = (image['image']/127.5 - 1.0).astype(np.float32)

                img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    z_q, emb_loss, perplexity, indices = vqgan_model.get_codebook_indices(img_tensor)
                # Store indices in the dictionary
                # print(i, file_path, file_path.split('/')[-1][:-4])
                self.indices_dict[file_path.split('/')[-1][:-4]] = indices.cpu().numpy().tolist()[0]
                print(f"Processed image {i + 1}/{len(file_paths)}")

            except Exception as e:
                print(f"Error processing image {i + 1}: {str(e)}")

        return indices
    # Function to load the saved model and decode images
    def decode_and_save_images(self, codebook_indices_path, save_folder):
        vqgan_model = VQGanVAE(vqgan_model_path=model_path, vqgan_config_path=config_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vqgan_model.to(device)
        vqgan_model.eval()

        # Decode images and save them to the specified folder
        for image_key, indices in self.indices_dict.items():
            indices = torch.tensor(indices).to(device)
            print(indices)
            decoded_image = vqgan_model.decode(indices)
            print(decoded_image.shape)
            decoded_image = decoded_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            decoded_image = (decoded_image * 255).astype(np.uint8)

            image_save_path = f"{save_folder}/{image_key}.png"
            Image.fromarray(decoded_image).save(image_save_path)
            print(f"Decoded image saved: {image_save_path}")
# Example usage:
if __name__ == "__main__":
    test_obj = YourTestClass()
    
    # Replace these paths with your actual paths
    model_path = '/share/ssddata/santosh_models/data_from_G42/temporal_project/final_data/CVPR/mimic_vqgan/last.ckpt'
    config_path = '/share/ssddata/santosh_models/data_from_G42/temporal_project/final_data/CVPR/mimic_vqgan/2021-12-17T08-58-54-project.yaml'
    image_paths_file = '/share/ssddata/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10999737/s52341872/5aea5877-40b40fee-5bccd163-ca1bf0ce-a95c213d.jpg'
    codebook_indices_path = './codebook_indices.pickle'


    save_folder = './decoded_images'

    # Process images and save the model
    indices = test_obj.process_images_and_save_model(model_path, config_path, image_paths_file)
    print(indices)
    img1 = indices[0]
    img1  = torch.cat([torch.tensor([1027], device='cuda:0'), img1])
    img1 = torch.cat([img1, torch.tensor([1028], device='cuda:0')])
    print(img1)
    model.load_state_dict(torch.load('/share/ssddata/santosh_models/data_from_G42/temporal_project/final_data/CVPR/temporal_project/trained_models/exp-5_best_model/epoch=164-train_loss= 4.39.ckpt')['state_dict'])
    model.to('cuda')
    model.eval()
    model.max_img_num = 1
    model.target_count = 1
    batch = {
        'img1': torch.tensor(indices),  # Convert to tensor if not already
        'txt': [],
        'modes': [],
        'view_position': [['AP_and_curr']], 
        'image_state': ['1'],
        'dt_in_days': torch.tensor([-1], device='cuda:0')
    }
    output = model.test_step(batch, batch_idx=0)
    gen_text = output['gen_text']
    gen_text_i = gen_text.tolist()
    gen_decoded_text_i = tokenizer.decode(gen_text_i[0], skip_special_tokens=True)
    print(gen_decoded_text_i)
    # test_obj.decode_and_save_images(codebook_indices_path=codebook_indices_path,save_folder=save_folder)

    # trainer = pl.Trainer(**trainer_args, logger=wandb_logger, plugins=DDPPlugin(find_unused_parameters=True), 
    #                         gradient_clip_val=args.gradient_clip_val, profiler="simple", limit_train_batches=0, limit_val_batches=0)
    # trainer.test(model, test_dataloaders=dm.test_dataloader())
