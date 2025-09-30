#manipolazione dati
import torch  
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import numpy as np
#sistama 
import os
import random
import argparse
import logging
#utility
from utils import utility_fun as uf

#dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from dataset import SalientDataset 

#modelli 
from model import  FixationPredictorV1, MultiTaskLoss,FixationPredictorV2,FixationPredictorV3

path_env_dict=uf.get_path_env()
#TRAIN/VAL/TEST PATH 
TEST_DATA_PATH = path_env_dict['TEST_DATA_PATH']

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_RESIZED_SHAPE = (224,224)

logging.getLogger().setLevel(logging.WARNING)
LOGGER_ =  logging.getLogger("TRAIN_LOGGER") 
LOGGER_.setLevel(logging.DEBUG)

if LOGGER_.hasHandlers():
    LOGGER_.handlers.clear()

stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
LOGGER_.addHandler(stream_handler)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def test(val_loader, model, device,case_is_multi_task):
    model.eval()
    MODEL_IS_MULTITASK = model.is_multi_task
    running_loss = 0.0

    eval_metric_dict={'auc_roc_continuous':0,'auc_roc_binary':0, 'auc_pr_continuous':0,'auc_pr_binary':0,'nss_continuous':0,'nss_binary':0, 'sim_continuous':0}

    all_continuous_preds = []
    all_continuous_for_binary_preds = []
    
    all_continuous_targets = []
    all_binary_targets = []
    with torch.no_grad():
        for inputs, continuous_target, binary_target  in val_loader:
            
            #print('binary target==>',inspect_binary_map(binary_target))
            USE_CONTINUOUS = False if isinstance(continuous_target,list) and len(continuous_target)==0 else True 
            inputs = inputs.to(device)
            continuous_target = continuous_target.to(device) if USE_CONTINUOUS==True else None
            binary_target = binary_target.to(device) 
            continuous_pred = None
            binary_pred = None
            
            first_map_out, second_map_out = model(inputs)
            if case_is_multi_task==True and MODEL_IS_MULTITASK==False:  
                raise 'Modello deve gestire entrambi i task per dati target binari e continui'
                sys.exit(-1)
            if case_is_multi_task==True: 
                continuous_pred = first_map_out
                binary_pred = second_map_out
            elif USE_CONTINUOUS==True: 
                continuous_pred = first_map_out
            elif  MODEL_IS_MULTITASK==False:
                binary_pred = first_map_out
            else:  
                binary_pred = second_map_out
            
            #Memorizza predizioni e i due target. Se i dati sono usecase =multitask oppure predizione continua allora ci aspettiamo
            #il  continuous_target esistente e lo useremo per le metriche che lavorano solo sul continuo (SIM, CORR).
            #Inoltre potremo sempre calcolare quelle comuni (AuC e NSS)
            if continuous_pred is not None:  
                #print('continuous pred==>',continuous_pred.cpu().shape)
                all_continuous_preds.append(torch.sigmoid(continuous_pred).cpu())
                all_continuous_targets.append(continuous_target.cpu())
            if binary_pred is not None:  
                #print('binary pred==>',binary_pred.cpu().shape)
                all_continuous_for_binary_preds.append(torch.sigmoid(binary_pred).cpu())
            
            #tutti i task binari o non e single task vs multi task prevedono sempre la possibilitù di 
            #computare metriche come Nss auc che richiedono binary mask 
            all_binary_targets.append(binary_target.cpu().float())

        # Concatenate tutti batches
        conc_continuous_preds = torch.cat(all_continuous_preds, dim=0).squeeze(1).numpy() if len(all_continuous_preds)>0 else [] # [N, H, W]
        conc_continuous_targets  = torch.cat(all_continuous_targets, dim=0).squeeze(1).numpy() if len(all_continuous_targets)>0 else []
        conc_continuous_for_binary_preds = torch.cat(all_continuous_for_binary_preds, dim=0).squeeze(1).numpy() if len(all_continuous_for_binary_preds)>0 else [] 
        conc_binary_targets = torch.cat(all_binary_targets, dim=0).squeeze(1).numpy() if len(all_binary_targets)>0 else []
        
        # --- AUC-ROC & PR ---
        # Valutazione rispetto al task di predizione mappa continua
        auc_scores_continuous_pred = []
        auc_pr_scores_continuous_pred=[]
        for map_true, map_est in zip(conc_binary_targets,conc_continuous_preds):
            auc_scores_continuous_pred.append(uf.compute_roc_auc(binary_map_true=map_true, gaussian_map_estimated=map_est))
            auc_pr_scores_continuous_pred.append(uf.compute_auc_pr(binary_map_true=map_true, gaussian_map_estimated=map_est))
        auc_scores_continuous_pred_MEAN = float(np.mean(auc_scores_continuous_pred)) if len(auc_scores_continuous_pred)>2 else []
        auc_pr_scores_continuous_pred_MEAN = float(np.mean(auc_pr_scores_continuous_pred)) if len(auc_pr_scores_continuous_pred)>2 else []

        #Valutazione rispetto al task di predizione mappa binaria
        auc_scores_continuous_for_binary_pred = []
        auc_pr_scores_continuous_for_binary_pred=[]
        for map_true, map_est in zip(conc_binary_targets,conc_continuous_for_binary_preds):
            auc_scores_continuous_for_binary_pred.append(uf.compute_roc_auc(binary_map_true=map_true, gaussian_map_estimated=map_est))
            auc_pr_scores_continuous_for_binary_pred.append(uf.compute_auc_pr(binary_map_true=map_true, gaussian_map_estimated=map_est))

        auc_scores_continuous_for_binary_pred_MEAN = float(np.mean(auc_scores_continuous_for_binary_pred)) if len(auc_scores_continuous_for_binary_pred)>2 else []
        auc_pr_scores_continuous_for_binary_pred_MEAN = float(np.mean(auc_pr_scores_continuous_for_binary_pred)) if len(auc_pr_scores_continuous_for_binary_pred)>2 else []

        eval_metric_dict['auc_roc_continuous'] = auc_scores_continuous_pred_MEAN  
        eval_metric_dict['auc_roc_binary'] = auc_scores_continuous_for_binary_pred_MEAN  
        eval_metric_dict['auc_pr_continuous'] = auc_pr_scores_continuous_pred_MEAN  
        eval_metric_dict['auc_pr_binary'] = auc_pr_scores_continuous_for_binary_pred_MEAN  
        
        # --- NSS------
        #valutazione rispetto task predizione mappa continua 
        nss_scores_continuous_pred = []
        for map_true, map_est in zip(conc_binary_targets, conc_continuous_preds):
            nss_scores_continuous_pred.append(uf.compute_nss(binary_map_true=map_true, gaussian_map_estimated=map_est))
        nss_scores_continuous_pred_MEAN = float(np.mean(nss_scores_continuous_pred)) if len(nss_scores_continuous_pred) > 2 else []
        #valutazione rispetto task predizione mappa binaria 
        nss_scores_continuous_for_binary_pred = []
        for map_true, map_est in zip(conc_binary_targets, conc_continuous_for_binary_preds):
            nss_scores_continuous_for_binary_pred.append(uf.compute_nss(binary_map_true=map_true, gaussian_map_estimated=map_est))
        nss_scores_continuous_for_binary_pred_MEAN = float(np.mean(nss_scores_continuous_for_binary_pred)) if len(nss_scores_continuous_for_binary_pred) > 2 else []
        eval_metric_dict['nss_continuous'] = nss_scores_continuous_pred_MEAN  
        eval_metric_dict['nss_binary'] = nss_scores_continuous_for_binary_pred_MEAN  
        #--- SIM (per il task su predizione mappa continua
        sim_scores = []
        for map_true, map_est in zip(conc_continuous_targets, conc_continuous_preds):
            sim_scores.append(uf.compute_similarity(map_true, map_est))
        sim_scores_MEAN = float(np.mean(sim_scores)) if len(sim_scores) > 2 else []
        eval_metric_dict['sim_continuous'] = sim_scores_MEAN  
       
        #print('auc_score continuous pred==>',auc_scores_continuous_pred_MEAN)
        #print('auc_score continuous _for_binary pred==>',auc_scores_continuous_for_binary_pred_MEAN)

    return eval_metric_dict
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
 
    #batch size
    parser.add_argument('--bs', type=int, default=32)
   
    parser.add_argument('--checkpoint_path',type=str,default='none')
    #salvataggio modello e metriche
    parser.add_argument('--model_path', type=str)
    #configurazione modello e dati
    parser.add_argument('--data_usecase', type=str)
    parser.add_argument('--use_multi_task_model',  type=str2bool)
    parser.add_argument('--stat_path',type=str)

    #sistema 
    parser.add_argument('--debug_mode',  type=str2bool,default=True)
    args, unknown = parser.parse_known_args()
    LOGGER_.debug(f'args==>{args}')
    if args.debug_mode: 
        LOGGER_.setLevel(logging.DEBUG)
    else: 
        LOGGER_.setLevel(logging.INFO)

    
    fixation_model = FixationPredictorV3(is_multi_task=args.use_multi_task_model)
    if args.checkpoint_path!= 'none':
        checkpoint = torch.load( args.checkpoint_path)
        fixation_model.load_state_dict(checkpoint,strict=False)
        LOGGER_.info(f'LOADED MODEL FROM CHECKPOINT:{args.checkpoint_path}')
    fixation_model=fixation_model.to(DEVICE)
    MODEL_ID = args.checkpoint_path.split('/')[-2]

    #---trasformazioni da applicare
    TRANSFORM_INPUT= transforms.Compose([
            transforms.Resize(FRAME_RESIZED_SHAPE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    TRANSFORM_OUPUT = transforms.Compose([
                 transforms.Resize(FRAME_RESIZED_SHAPE,interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor()])
    TRANSFORM_BINARY_OUTPUT = transforms.Compose([
                transforms.ToTensor()])

    #defini lista video da usare nel test set 
    test_data_paths=uf.read_json(path=os.path.join(TEST_DATA_PATH,'test'+'.json'))
    video_id_list=np.unique(list(map(lambda x:  int(x[0].split('/')[-2]),test_data_paths['usecase'][args.data_usecase])))
    model_stat={'video_id':[],'nss_binary':[],'nss_continuous':[],'nss_binary_multi':[],'nss_continuous_multi':[],  'sim_continuous':[],'sim_continuous_multi':[]}
    for video_id in video_id_list:
        LOGGER_.info(f'### TESTING ON VIDEO WITH ID: {video_id} #####')
        test_loader = DataLoader(
                        SalientDataset(source_path=TEST_DATA_PATH,dataset_mode='test',usecase=args.data_usecase,binary_target_transform=TRANSFORM_BINARY_OUTPUT,target_transform=TRANSFORM_OUPUT,transform =TRANSFORM_INPUT,video_id=video_id ),
                        batch_size=args.bs, shuffle=False)
        
        test_metric_dict = test(test_loader, fixation_model, DEVICE,case_is_multi_task=False)
        expected_nss_value = test_metric_dict.get('nss_binary',0) if 'continuous' not in args.data_usecase else test_metric_dict.get('nss_continuous',0)

        expcted_case = 'binary' if 'continuous' not in args.data_usecase else 'continuous'
        if args.use_multi_task_model==True:  
            #valutazione rispetto  target binario
            if 'continuous' not in args.data_usecase:
                model_stat['nss_binary_multi'].append(expected_nss_value) 
            else:
                model_stat['nss_continuous_multi'].append(expected_nss_value)
                model_stat['sim_continuous_multi'].append(test_metric_dict.get('sim_continuous',0))
        elif 'continuous' not in args.data_usecase:
            model_stat['nss_binary'].append(expected_nss_value) 
        else:
            model_stat['nss_continuous'].append(expected_nss_value)
            model_stat['sim_continuous'].append(test_metric_dict.get('sim_continuous',0))
                
        model_stat['video_id'].append(int(video_id))
        LOGGER_.info(f'TEST LEN=>{len(test_loader.dataset)}')
        eval_stat_path = os.path.join( args.stat_path, f"model_modelName_{fixation_model.model_name}_{'test'}.json")
        uf.write_json(data=model_stat,path=eval_stat_path)
    