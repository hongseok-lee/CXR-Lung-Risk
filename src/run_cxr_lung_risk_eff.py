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
from fastai.vision.all import *
from sklearn.metrics import *
from pathlib import Path
from fastai.vision.all import *
from torch.utils.data import DataLoader
import torch
import pretrainedmodels
import pandas as pd
import fastai
from tqdm import tqdm


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
    ensemble_weights_fn = config["ensemble_weights_fn"]
    model_details_fn = config["model_details_fn"]
    mdl_dir = config["mdl_dir"]
    mdl_name = config["mdl_name"]

    use_gpu = config["use_gpu"]
    gpu_id = config["gpu_id"]

    test_set_dir = config["test_set_dir"]
    test_dataset_name = config["test_dataset_name"]

    out_file_path = config["out_file_path"]

    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


    model_details_df = pd.read_csv(model_details_fn)
    ensemble_weights_df = pd.read_csv(ensemble_weights_fn)

    patients_list = [f for f in os.listdir(test_set_dir) if os.path.isfile(os.path.join(test_set_dir, f))] 
      # The results of the inference phase are stored in the DataFrame "results_df"

    # Dummy is a dummy nonsense variable to act as the fake "target variable" - necessary for the pipeline to run
    # The column "valid_col" is True for all samples except for an artificial sample at the end
    # (since for the fast.ai learner to work, there needs to be a "training set" included too)
    output_df = pd.DataFrame(columns = ['File', 'Dummy', 'Prediction'])
    output_df['File'] = patients_list
    output_df['Dummy'] = np.random.random_sample(len(patients_list))
    output_df['valid_col'] = np.repeat(True, output_df.shape[0])

    # Add an additional image to act as the dummy training set, by setting the "valid_col" value to False
    results_df = output_df.append(output_df.iloc[output_df.shape[0] -1, :],
                                    ignore_index = True)

    results_df.loc[results_df.shape[0] -1, 'valid_col'] = False  

    # The number of models in the ensemble corresponds to the number of rows in the "model_details_df" dataframe


    
    
    device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")
    
    ensemble_weights = pd.read_csv(config["ensemble_weights_fn"])["weight"].values
    
    
    model_number = model_details_df.shape[0]
    pred_arr = np.zeros((results_df.shape[0]-1, model_number))
    
    
    

    class EnsembleModel(nn.Module):
        def __init__(self, models):
            super().__init__()
            self.models = nn.ModuleList(models)
        
        def forward(self, x):
            outputs = [model(x) for model in self.models]
            return torch.stack(outputs)

    def create_ensemble_dataloader(size_to_ds, bs=64):
        size_to_dl = {}
        for s, ds in size_to_ds.items():
            size_to_dl[s] = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)
        return size_to_dl 

    def ensemble_predict(size_to_models, size_to_dl):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        all_preds = []
        all_labels = []
        size_to_ensemble = {} 
        for s in [224]:
            models = size_to_models[s]
            ensemble = EnsembleModel(models).to(device)
            ensemble.eval()
            size_to_ensemble[s] = ensemble
             
        imgs = ImageDataLoaders.from_df(df = results_df, path = test_set_dir,
                                label_col = "Dummy", y_block = RegressionBlock, bs = 4,
                                val_bs = 4, valid_col = "valid_col",
                                item_tfms = Resize(224),
                                batch_tfms = [Normalize.from_stats(*imagenet_stats)])
        size_to_preds = {} 
        with torch.no_grad():
            for s in [224]:
                dl = size_to_dl[s]
                ensemble = size_to_ensemble[s]
                preds_list = []
                for batch in tqdm(imgs.valid):
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    outputs = ensemble(inputs)
                    preds = outputs
                    preds_list.append(preds.cpu().numpy())
                size_to_preds[s] = preds_list
        
        return size_to_preds

    def get_model(model_id, model_details_df, out_nodes):
        imgs = ImageDataLoaders.from_df(df = results_df, path = test_set_dir,
                                      label_col = "Dummy", y_block = RegressionBlock, bs =4,
                                      val_bs = 4, valid_col = "valid_col",
                                      item_tfms = Resize(16),
                                      batch_tfms = [Normalize.from_stats(*imagenet_stats)])
        model_arch = model_details_df.Architecture[model_id].lower()
        
        if model_arch == "inceptionv4":
            def _get_model(pretrained=True, model_name='inceptionv4', **kwargs):
                if pretrained:
                    arch = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
                else:
                    arch = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
                return arch
            def get_cadene_model(pretrained=True, **kwargs):
                return fastai_inceptionv4
            custom_head = create_head(nf=2048*2, n_out=out_nodes)
            fastai_inceptionv4 = nn.Sequential(*list(_get_model(model_name='inceptionv4').children())[:-2], custom_head)
            return cnn_learner(imgs , get_cadene_model, n_out=out_nodes)
        
        elif model_arch == "resnet34":
            return cnn_learner(imgs, fastai.vision.models.resnet34, n_out=out_nodes)
        
        elif model_arch == "tiny":
            mdl = SimpleArchs.get_simple_model("Tiny", out_nodes)
            return Learner(imgs, mdl)
        
        else:
            return cnn_learner(imgs, mdl, n_out = out_nodes)
            # raise ValueError(f"Architecture type: {model_arch} not supported. Please make sure the `model_spec` CSV is found in the working directory and can be accessed.")

    # 사용 예시
    model_details_df = pd.read_csv(model_details_fn)  # 모델 사양이 저장된 CSV 파일
    out_nodes = 1  # 출력 클래스 수 # csv 에 다 1임. 

    models = []
    size_to_models = {
        224:[]
    }
    for model_id in range(len(model_details_df)):
        size = int(model_details_df.Image_Size[model_id])
        learn = get_model(model_id, model_details_df, out_nodes)
        # learn.load(f'model_{model_id}.pth')  # 각 모델의 가중치 로드
        
        learn.path = Path(mdl_dir.split("models")[0])
        learn.load(mdl_name + "_" + str(model_id))
        size_to_models[size].append(learn.model)
        models.append(learn.model)

    # 데이터 로더 생성 (실제 데이터셋으로 대체 필요)
    # dls = DataLoaders(train=None, valid=None)  # 실제 데이터셋으로 초기화 필요
    sizes = [
        224
            ]
    
    size_to_ds = {} 
    bs,val_bs = 4,4
    
    for s in sizes:
        dls = ImageDataLoaders.from_df(df = results_df, path = test_set_dir,
                                        label_col = "Dummy", y_block = RegressionBlock, bs = bs,
                                        val_bs = val_bs, valid_col = "valid_col",
                                        item_tfms = Resize(size),
                                        batch_tfms = [Normalize.from_stats(*imagenet_stats)])
        size_to_ds[s] = dls.valid.dataset
    
 
    
    
    size_to_dl = create_ensemble_dataloader(size_to_ds)

    size_to_pred = ensemble_predict(size_to_models, size_to_dl)
    
    predictions = size_to_pred[224]
    for p in predictions:
        print(p)
    # results = []
    # for batch in predictions:
    #     result = np.stack(batch, axis=1)
    #     print(result.shape)
    #     results.append(result)
    # predictions = np.concatenate(results, axis=0)
    
        
    # predictions = np.concatenate(predictions, axis=0)[:,1]
    
    print(predictions.shape) 
    
    lasso_intercept = 49.8484258
    predictions = np.matmul(pred_arr, ensemble_weights) + lasso_intercept
    print(predictions)
    
    output_df['CXR_Lung_Risk'] = predictions
    output_df = output_df.drop(["valid_col", "Dummy", "Prediction"], axis = 1)

    output_df.to_csv(out_file_path, index = False)

if __name__ == "__main__":
    
  parser = argparse.ArgumentParser(description = 'Run the CXR-Lung-Risk inference pipeline.')

  parser.add_argument("--conf",
                      required = False,
                      help = "Specify the path to the YAML file containing details for the inference phase. " \
                             "Tries to default to 'config.yaml' under the 'src/' directory.",
                      default = "config.yaml")

  args = parser.parse_args()

  conf_file_path = os.path.join(args.conf)

  with open(conf_file_path) as f:
    yaml_conf = yaml.load(f, Loader = yaml.FullLoader)

  # dict storing the config args needed to run the main function
  config = dict()

  # path to the directory storing the test set
  _, test_dataset_name = os.path.split(yaml_conf["input"]["test_set_dir"])
  config["test_set_dir"] = yaml_conf["input"]["test_set_dir"]
  config["test_dataset_name"] = test_dataset_name

  # path to the directory storing the models
  config["mdl_dir"] = "../models"

  # name of the CSV file storing the models details (e.g., architecture)
  config["model_details_fn"] = "CXR_Lung_Risk_Specs.csv"
  
  # base name for the ".pth" files of all the models in the ensemble
  config["mdl_name"] = "Lung_Age_081221"

  # path to the directory where the output should be stored, and base file name for the output
  out_base_path = yaml_conf["output"]["out_base_path"]

  if not os.path.exists(out_base_path):
    os.mkdir(out_base_path)

  out_fn = "cxr_lung_risk_" + test_dataset_name + ".csv"

  config["out_file_path"] = os.path.join(out_base_path, out_fn)

  # name of the CSV file storing the weights for the ensemble model
  config["ensemble_weights_fn"] = "ensemble_weights.csv"

  # whether to use the GPU for the processing or not
  config["use_gpu"] = yaml_conf["processing"]["use_gpu"] 
  config["gpu_id"] = yaml_conf["processing"]["gpu_id"] 

  working_dir = os.getcwd()

  # check the script is running from the source code directory of the repository
  print("Current working directory:", working_dir)
  
  assert(os.path.exists(config["mdl_dir"]) and
         os.path.exists(config["model_details_fn"]) and
         os.path.exists(config["test_set_dir"]) and 
         os.path.exists(config["ensemble_weights_fn"])
         )

  assert(len(os.listdir(config["mdl_dir"])) == 20 and
         len(os.listdir(config["test_set_dir"])) > 0
         )

  print("Location to be parsed for images to process:", config["test_set_dir"])
  print("Location where the output should be saved at:", config["out_file_path"])

  run_cxr_lung_risk(config)