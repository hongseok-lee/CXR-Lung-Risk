import os
import argparse
import yaml
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pretrainedmodels
import SimpleArchs
from concurrent.futures import ThreadPoolExecutor

warnings.simplefilter(action='ignore')

class CXRDataset(Dataset):
    def __init__(self, image_dir, file_list, transform):
        self.image_dir = image_dir
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

def load_model(model_arch, out_nodes):
    if model_arch == "inceptionv4":
        model = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')
        custom_head = create_head(nf=2048*2, n_out=37)
        return nn.Sequential(*list(model.children())[:-2], custom_head)
    elif model_arch == "resnet34":
        return fastai.vision.models.resnet34
    elif model_arch == "tiny":
        return SimpleArchs.get_simple_model("Tiny", out_nodes)
    else:
        raise ValueError(f"Unsupported architecture: {model_arch}")

def run_inference(model, dataloader, device):
    model.eval()
    predictions = []
    file_names = []
    with torch.no_grad():
        for batch, names in dataloader:
            batch = batch.to(device)
            output = model(batch)
            predictions.extend(output.cpu().numpy()[:, 0])
            file_names.extend(names)
    return predictions, file_names

def run_cxr_lung_risk(config):
    device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")
    
    model_details_df = pd.read_csv(config["model_details_fn"])
    ensemble_weights = pd.read_csv(config["ensemble_weights_fn"])["weight"].values
    
    patients_list = [f for f in os.listdir(config["test_set_dir"]) if os.path.isfile(os.path.join(config["test_set_dir"], f))]
    
    pred_arr = np.zeros((len(patients_list), len(model_details_df)))
    
    # Pre-load all models
    models = {}
    for model_id, model_row in model_details_df.iterrows():
        model = load_model(model_row.Architecture.lower(), int(model_row.Num_Classes))
        model.load_state_dict(torch.load(os.path.join(config["mdl_dir"], f"{config['mdl_name']}_{model_id}.pth"), map_location=device))
        model.to(device).eval()
        models[model_id] = model

    for model_id, model_row in model_details_df.iterrows():
        size = int(model_row.Image_Size)
        normalize = int(model_row.Normalize) == 1
        
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(*imagenet_stats) if normalize else lambda x: x
        ])
        
        dataset = CXRDataset(config["test_set_dir"], patients_list, transform)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
        
        predictions, file_names = run_inference(models[model_id], dataloader, device)
        
        for i, file_name in enumerate(file_names):
            idx = patients_list.index(file_name)
            pred_arr[idx, model_id] = predictions[i]
    
    lasso_intercept = 49.8484258
    predictions = np.matmul(pred_arr, ensemble_weights) + lasso_intercept
    
    output_df = pd.DataFrame({'File': patients_list, 'CXR_Lung_Risk': predictions})
    output_df.to_csv(config["out_file_path"], index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the CXR-Lung-Risk inference pipeline.')
    parser.add_argument("--conf", required=False, help="Path to the YAML config file.", default="config.yaml")
    args = parser.parse_args()
    
    with open(args.conf) as f:
        yaml_conf = yaml.safe_load(f)
    
    config = {
        "test_set_dir": yaml_conf["input"]["test_set_dir"],
        "mdl_dir": "../models",
        "model_details_fn": "CXR_Lung_Risk_Specs.csv",
        "mdl_name": "Lung_Age_081221",
        "out_file_path": os.path.join(yaml_conf["output"]["out_base_path"], f"cxr_lung_risk_{os.path.basename(yaml_conf['input']['test_set_dir'])}.csv"),
        "ensemble_weights_fn": "ensemble_weights.csv",
        "use_gpu": yaml_conf["processing"]["use_gpu"],
        "gpu_id": yaml_conf["processing"]["gpu_id"]
    }
    
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_id"] if config["use_gpu"] else ""
    
    run_cxr_lung_risk(config)