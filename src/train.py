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
TRAIN_DATA_PATH = path_env_dict['TRAIN_DATA_PATH']
VAL_DATA_PATH = path_env_dict['VAL_DATA_PATH']
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
def train_and_validate(train_loader,val_loader,model,args):

    MODEL_ARTIFACT_PATH= args.model_path
    EPOCH_CNT = args.epochs
    EPOCH_START = args.epoch_start
    IS_DATA_FOR_MULTI_TASK = args.is_data_for_multi_task
    LEARNING_RATE = args.lr

    WEIGH_LOSS_BINARY = args.binary_loss_w
    WEIGH_LOSS_CONTINUOUS = args.continuous_loss_w
    
    #stima peso pixel "attivi" (>0). 
    BINARY_POS_WEIGTH = 1 
    CONTINUOUS_POS_WEIGHT = 1
    used_binary = None
    
    #circa 40 frame per video
    SUBSET_SIZE = 2500
    all_indices = list(range(len(train_loader.dataset)))
    subset_indices = random.sample(all_indices, SUBSET_SIZE)
    TRAIN_SUBSET = Subset(train_loader.dataset, subset_indices)
    if IS_DATA_FOR_MULTI_TASK==True:  
        binary_pos_w_est,_ = uf.estimate_pos_weight(TRAIN_SUBSET,use_binary=True)
        BINARY_POS_WEIGTH = min([binary_pos_w_est, np.sqrt(binary_pos_w_est)])
        continuous_pos_w_est,_ = uf.estimate_pos_weight(TRAIN_SUBSET,use_binary=False)
        CONTINUOUS_POS_WEIGHT = min([continuous_pos_w_est, np.sqrt(continuous_pos_w_est**2)])
    else: 
        _pos_w_est,used_binary = uf.estimate_pos_weight(TRAIN_SUBSET,use_binary=None)
        if used_binary: 
            BINARY_POS_WEIGTH = min([_pos_w_est, np.sqrt(_pos_w_est)])
        else:
            CONTINUOUS_POS_WEIGHT = min([_pos_w_est, np.sqrt(_pos_w_est**2)])
        
    LOGGER_.debug(f'USED BINARY: {used_binary} BINARY_POS_WEIGTH=>{BINARY_POS_WEIGTH} CONTINUOUS_POS_WEIGHT=>{CONTINUOUS_POS_WEIGHT}')
    LOGGER_.debug(f'WEIGHTED LOSS :  BINARY_LOSS_WEIGTH=>{WEIGH_LOSS_BINARY} CONTINUOUS_LOSS_WEIGHT=>{WEIGH_LOSS_CONTINUOUS}')
    
    criterion = MultiTaskLoss(weight_binary=WEIGH_LOSS_BINARY, weight_continuous=WEIGH_LOSS_CONTINUOUS,binary_loss_pos_weight=BINARY_POS_WEIGTH,continuous_loss_pos_weight=CONTINUOUS_POS_WEIGHT).to(DEVICE)
    MODEL_NAME = model.model_name

    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True,
        min_lr=1e-6
    )

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    all_metric_hist_dict={'binary_pos_weight':BINARY_POS_WEIGTH, 'continuous_pos_weight':CONTINUOUS_POS_WEIGHT, 'binary_loss_weight':WEIGH_LOSS_BINARY, 'continuous_loss_weight':WEIGH_LOSS_CONTINUOUS,
                          'start_epoch':0,'end_epoch': EPOCH_CNT, 'last_done_epoch':0,'train_loss':[] ,'eval_loss':[],'auc_roc_continuous':[],'auc_pr_continuous':[],'auc_pr_binary':[],'auc_roc_binary':[], 'nss_continuous':[],'nss_binary':[], 'sim_continuous':[]}
    
    for epoch in range(EPOCH_START,EPOCH_CNT):
        train_loss = train(train_loader, model, criterion, optimizer, epoch, DEVICE,case_is_multi_task=IS_DATA_FOR_MULTI_TASK)
        val_metric_dict = validate(val_loader, model, criterion, DEVICE,case_is_multi_task=IS_DATA_FOR_MULTI_TASK)

        all_metric_hist_dict['train_loss'].append(train_loss)
        all_metric_hist_dict['last_done_epoch'] = epoch
        #LOGGER_.debug(f'all_metric_hist_dict==>{all_metric_hist_dict}')
        for k_ in val_metric_dict.keys():
            all_metric_hist_dict[k_].append(val_metric_dict[k_])
            
        val_loss = val_metric_dict['eval_loss']
        scheduler.step(val_loss) 
        if val_metric_dict['eval_loss'] <=best_val_loss -.0001:
            best_val_loss = val_metric_dict['eval_loss']
            model_path = os.path.join(MODEL_ARTIFACT_PATH, f"model_epoch_{epoch}_modelName_{MODEL_NAME}.pt")
            eval_stat_path = os.path.join(MODEL_ARTIFACT_PATH, f"model_epoch_{epoch}_modelName_{MODEL_NAME}_mode_{'validation'}.json")
            torch.save(model.state_dict(), model_path)
            #salva anche dizionario 
            #LOGGER_.debug(f'all_metric_hist_dict==>{all_metric_hist_dict}')
            uf.write_json(data=all_metric_hist_dict,path=eval_stat_path)
            LOGGER_.info(f'Best model at epoch {epoch} with val loss: {best_val_loss:.4f}')
    

def train(train_loader, model, criterion, optimizer, epoch, device, case_is_multi_task):
    model.train()
    running_loss = 0.0
    MODEL_IS_MULTITASK = model.is_multi_task
    for i, (inputs, continuous_target, binary_target) in enumerate(train_loader):        
        USE_CONTINUOUS = False if isinstance(continuous_target,list) and len(continuous_target)==0 else True 
        inputs = inputs.to(device)
        continuous_target = continuous_target.to(device) if USE_CONTINUOUS==True else None
        binary_target = binary_target.to(device) 
        continuous_pred = None
        binary_pred = None
        optimizer.zero_grad()
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
            
        
        loss = criterion(binary_pred=binary_pred, binary_target=binary_target, continuous_pred=continuous_pred, continuous_target=continuous_target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        if i % 10 == 0:
            LOGGER_.info(f'[Train] Epoch {epoch} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader.dataset)
    LOGGER_.info(f'[Train] Epoch {epoch} Summary — Loss: {epoch_loss:.4f}')
    return epoch_loss
def validate(val_loader, model, criterion, device,case_is_multi_task):
    model.eval()
    MODEL_IS_MULTITASK = model.is_multi_task
    running_loss = 0.0

    eval_metric_dict={'eval_loss':100,'auc_roc_continuous':0,'auc_roc_binary':0, 'auc_pr_continuous':0,'auc_pr_binary':0,'nss_continuous':0,'nss_binary':0, 'sim_continuous':0}

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
            
            loss = criterion(binary_pred=binary_pred, binary_target=binary_target, continuous_pred=continuous_pred, continuous_target=continuous_target)
            running_loss += loss.item() * inputs.size(0)

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

    epoch_loss = running_loss / len(val_loader.dataset)
    eval_metric_dict['eval_loss'] = epoch_loss  
    
    LOGGER_.info(f'[Val] Loss: {epoch_loss:.4f}')
    return eval_metric_dict
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    #parametri training 
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    #batch size
    parser.add_argument('--bs', type=int, default=32)

    #utile per ripartire da checkpoint
    parser.add_argument('--epoch_start', type=int, default=0)
    parser.add_argument('--checkpoint_path',type=str,default='none')
    #salvataggio modello e metriche
    parser.add_argument('--model_path', type=str)
    #configurazione modello e dati
    parser.add_argument('--model_version', type=str, default='v1')
    parser.add_argument('--data_usecase', type=str)
    parser.add_argument('--use_multi_task_model',  type=str2bool)
    parser.add_argument('--binary_loss_w',type=float, default=0.5)
    parser.add_argument('--continuous_loss_w',type=float, default=0.5)

    parser.add_argument('--is_data_for_multi_task', type=str2bool)
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
    
    train_loader = DataLoader(
                    SalientDataset(source_path=TRAIN_DATA_PATH,dataset_mode='train',usecase=args.data_usecase,target_transform=TRANSFORM_OUPUT,binary_target_transform=TRANSFORM_BINARY_OUTPUT,transform =TRANSFORM_INPUT ),
                    batch_size=args.bs, shuffle=True)
    LOGGER_.info(f'TRAIN LEN=>{len(train_loader.dataset)}')
    val_loader = DataLoader(
                    SalientDataset(source_path=VAL_DATA_PATH,dataset_mode='val',usecase=args.data_usecase,binary_target_transform=TRANSFORM_BINARY_OUTPUT,target_transform=TRANSFORM_OUPUT,transform =TRANSFORM_INPUT ),
                    batch_size=args.bs, shuffle=True)
    LOGGER_.info(f'VAL LEN=>{len(val_loader.dataset)}')
 
    train_and_validate(train_loader,val_loader,fixation_model,args)
  