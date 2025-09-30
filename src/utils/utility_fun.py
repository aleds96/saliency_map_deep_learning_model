#manipolazione dati 
import cv2
import numpy as np 
import torch
import json
from scipy.stats import multivariate_normal
#sistema
import os
import gc

#displaying a video 
from IPython.display import HTML,display, clear_output
#loading mat
from scipy.io import loadmat
import mat73

#visualizzazione
import matplotlib.pyplot as plt
from PIL import Image
#valutazione 
from sklearn.metrics import roc_auc_score, roc_curve,precision_recall_curve,auc

def get_path_env():
    RAW_DB_PATH = os.path.join('dataset','raw')
    MODEL_DB_PATH = os.path.join('dataset','model')

    TRAIN_DATA_PATH = os.path.join(MODEL_DB_PATH,'train')
    VAL_DATA_PATH = os.path.join(MODEL_DB_PATH,'val')
    TEST_DATA_PATH = os.path.join(MODEL_DB_PATH,'test')

    RAW_FIX_DURATION_PATH = os.path.join(RAW_DB_PATH,'fix_duration_custom')
    RAW_FACIAL_MARK_PATH =  os.path.join(RAW_DB_PATH,'facial_mark_custom')
    GAUSSIAN_TARGET_PATH=os.path.join(RAW_DB_PATH,'target_data','gaussian_map_target')
    WEIGHT_GAUSSIAN_TARGET_PATH=os.path.join(RAW_DB_PATH,'target_data','weight_gaussian_map_target')
    BINARY_TARGET_PATH=os.path.join(RAW_DB_PATH,'target_data','binary_map_target')
    BINARY_TARGET_RES224_PATH=os.path.join(RAW_DB_PATH,'target_data','binary_map_target_res224')
    GAZE_MODEL_INPUT_EXAMPLE = os.path.join(RAW_DB_PATH,'example','gaze_model_input_sample')
    GAZE_INPUT_POINT_DATA_PATH = os.path.join(RAW_DB_PATH,'input_data','gaze_point_data')
    GAUSSIAN_ISO_INPUT_DATA_PATH=os.path.join(RAW_DB_PATH,'input_data','gaussian_isotropic_data')
    GAUSSIAN_COV_INPUT_DATA_PATH=os.path.join(RAW_DB_PATH,'input_data','gaussian_cov_data')
    GAUSSIAN_ISOk_INPUT_DATA_PATH=os.path.join(RAW_DB_PATH,'input_data','gaussian_isotropic_data_k')
    GAUSSIAN_COVk_INPUT_DATA_PATH=os.path.join(RAW_DB_PATH,'input_data','gaussian_cov_data_k')

    
    
    UNIMI_DB_PATH = os.path.join('find_dataset','Our_database') 
    FACIAL_MARKS= os.path.join('find_dataset','annotated_facial_landmarks')
    RAW_VIDEO= os.path.join(UNIMI_DB_PATH,'raw_videos') 
    FIX_DATA= os.path.join(UNIMI_DB_PATH,'fix_data')
    FIX_DATA_NEW= os.path.join(UNIMI_DB_PATH,'fix_data_NEW')
    FIX_DURATION_DATA = os.path.join(UNIMI_DB_PATH,'fix_and_dur_data')
    
    return {
        #custom db 
        "RAW_DB_PATH":RAW_DB_PATH,
        "TRAIN_DATA_PATH": TRAIN_DATA_PATH,
         "VAL_DATA_PATH": VAL_DATA_PATH,
        "TEST_DATA_PATH": TEST_DATA_PATH,
        "RAW_FIX_DURATION_PATH": RAW_FIX_DURATION_PATH, 
        "GAUSSIAN_TARGET_PATH":GAUSSIAN_TARGET_PATH,
        "WEIGHT_GAUSSIAN_TARGET_PATH":WEIGHT_GAUSSIAN_TARGET_PATH,
        "BINARY_TARGET_PATH":BINARY_TARGET_PATH,
        "BINARY_TARGET_RES224_PATH": BINARY_TARGET_RES224_PATH,
        "RAW_FACIAL_MARK_PATH":RAW_FACIAL_MARK_PATH,
        "GAZE_INPUT_EXAMPLE_PATH": GAZE_MODEL_INPUT_EXAMPLE,
        "GAZE_INPUT_POINT_DATA_PATH":GAZE_INPUT_POINT_DATA_PATH,
        "GAUSSIAN_ISO_INPUT_DATA_PATH":GAUSSIAN_ISO_INPUT_DATA_PATH,
        "GAUSSIAN_COV_INPUT_DATA_PATH": GAUSSIAN_COV_INPUT_DATA_PATH,
        "GAUSSIAN_COVk_INPUT_DATA_PATH":GAUSSIAN_COVk_INPUT_DATA_PATH,
        "GAUSSIAN_ISOk_INPUT_DATA_PATH":GAUSSIAN_ISOk_INPUT_DATA_PATH,
        
        #unimi db
        "UNIMI_DB_PATH": UNIMI_DB_PATH,
        "FACIAL_MARKS_UNIMI": FACIAL_MARKS,
        "RAW_VIDEO": RAW_VIDEO,
        "FIX_DATA": FIX_DATA,
        "FIX_DATA_NEW":FIX_DATA_NEW,
        "FIX_DURATION_DATA_UNIMI": FIX_DURATION_DATA
    }
def load_mat_data(path): 
    print('****reading new mat data')
    format_=0
    try: 
       out= loadmat(path)
    except Exception as e:  
        print('cannot read using loadmat fun',e)  
        print('trying to use mat73...')
        out = mat73.loadmat(path) 
        format_=1
    if out:  
        print('reading mat success!********')
    else: 
        print('reading mat failed!****')
    return out,format_
def display_video(video_path,w=400):
    if os.path.exists(video_path):
        mp4 = open(video_path, 'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        display(HTML(f"""
        <video width={w} controls>
            <source src="{data_url}" type="video/mp4">
        </video>
        """))
    else:
        print(f"Il file {video_path} non è stato trovato.")
def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()
def read_specific_frame(video_path,frame_idx): 
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    frame_out=None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Could not read frame {frame_idx}")
        return None
    return frame
def draw_landmarks(frame, landmarks, style='x'):
    
    for (x, y) in landmarks:
        x_=int(x)
        y_=int(y) 
        rev=False 
        if rev:
            tmp_= x_ 
            x_=y_ 
            y_= tmp_
        if style == 'circle':
            cv2.circle(frame, (x_, y_), 2, (0, 255, 0), -1)
        elif style == 'x':
            cv2.line(frame, (x_ - 2, y_ - 2), (x_ + 2, y_ + 2), (0, 0, 255), 1)
            cv2.line(frame, (x_ - 2, y_ + 2), (x_ + 2, y_ - 2), (0, 0, 255), 1)
    return frame
def show_frame_in_notebook(frame):
    #Mostra frame in Jupyter Notebook
    clear_output(wait=True)
    display(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    gc.collect()  
def display_video_with_landmarks(video_path, landmarks_per_frame, style='x'):
    for i, frame in enumerate(read_video_frames(video_path)):
        landmarks = landmarks_per_frame.get(i, [])
        frame = draw_landmarks(frame, landmarks, style)
        show_frame_in_notebook(frame)
    cv2.destroyAllWindows()
def resize_frame(frame, scale=0.5):
    return cv2.resize(frame, (0, 0), fx=scale, fy=scale)
def get_face_bounding_box(landmarks,pad_w_r=0.2,pad_h_r=0.2):
    #restituisce bounding box (x, y, w, h) dai facial landmarks
    if not landmarks:
        return None  # Handle empty input
    x_coords = [x for x, y in landmarks]
    y_coords = [y for x, y in landmarks]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x,y,w,h=pad_box(x=x_min, y=y_min, w=x_max - x_min, h=y_max - y_min,pad_w_r=pad_w_r,pad_h_r=pad_h_r)
   
    return  x,y,w,h
    
def pad_box(x, y, w, h, pad_w_r=0.2,pad_h_r=0.2):
    pad_w = int(w * pad_w_r)
    pad_h = int(h * pad_h_r)
    return (x - pad_w, y - pad_h, w + 2 * pad_w, h + 2 * pad_h)
#funzioni per accedere ai punti di fixation
#ottieni punti di fixation rispetto ad una coppia (observer, frame)
def get_fixation_points_per_obs_frame(fix_data, obs_id, frame_idx): 
    raw_data = fix_data['curr_v_all_s'][:,0]
    observer_count = len(raw_data) 
    if observer_count<1 or obs_id>observer_count-1:  
        print('getting fixation points not possible per obs',obs_id,'/',observer_count-1) 
        return None   
    frame_data = raw_data[obs_id]
    frame_count = frame_data.shape[0] 
    if frame_count<1 or frame_idx>frame_count-1: 
        print('getting fixation points not possible per frame',frame_idx,'/',frame_count)  
        return None   
    fix_points = frame_data[frame_idx,:]
    return (fix_points,observer_count, frame_count)
#ottieni punti di fixation per ogni frame rispetto ad un observer 
def get_fixation_points_per_obs(fix_data, obs_id): 
    raw_data = fix_data['curr_v_all_s'][:,0]
    observer_count = len(raw_data) 
    if observer_count<1 or obs_id>observer_count-1:  
        print('getting fixation points not possible per obs',obs_id,'/',observer_count-1) 
        return None   
    fix_points_per_frame = raw_data[obs_id]
    frame_count = fix_points_per_frame.shape[0] 
    return (fix_points_per_frame, observer_count,frame_count)
#ottieni info per fixdata
def get_fixation_points_info(fix_data): 
    raw_data = fix_data['curr_v_all_s'][:,0]
    observer_count = len(raw_data) 
    frame_count=0
    if observer_count>0:
        frame_count=raw_data[0].shape[0] 
    return observer_count,frame_count
##funzioni per accedere dati facciali 
def get_facial_mark_info_from_raw_data(facial_raw,format_mat):  
    if format_mat==1:  
        frame_count= len(facial_raw)
        max_faces_cnt=0
        for fid in range(frame_count):
            face_counts_tmp = len(facial_raw[fid][0]['shapes']['shape'])
            if face_counts_tmp>max_faces_cnt: 
                max_faces_cnt= face_counts_tmp  
        return frame_count,max_faces_cnt
    else: 
        frame_count=facial_raw.shape[0] 
        max_faces_cnt=0
        for fid in range(frame_count):
            face_counts_tmp = facial_raw[fid,0][0][0][0].shape[1]
            if face_counts_tmp>max_faces_cnt: 
                max_faces_cnt= face_counts_tmp  
        return frame_count,max_faces_cnt
def get_facial_mark_from_raw_data(facial_raw,format_mat,frame_id,face_id):  
    if format_mat==1:  
        return facial_raw[frame_id][0]['shapes']['shape'][face_id]
    return facial_raw[frame_id,0][0][0][0][0,face_id][0]
def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
def make_binary_mask_for_fixdata(x, y, my_size):
    
    #Crea una maschera binaria di fissazione.    
    
    R, C = my_size
    mask = np.zeros((R, C), dtype=np.uint8)

    for i in range(len(x)):
        temp_x = int(np.floor(x[i]))
        temp_y = int(np.floor(y[i]))

        # Verifica che il punto sia dentro i limiti dell'immagine
        if 0 <= temp_x < C and 0 <= temp_y < R:
            mask[temp_y, temp_x] = 1

    return mask

def make_gauss_mask(x, y, my_size, weight_in=None, sigma_in=None):
    R, C = my_size
    if weight_in is None:
        weight_in = np.ones_like(x)

    # Create grid of coordinates
    X, Y = np.meshgrid(np.arange(C), np.arange(R))
    pos = np.stack([X, Y], axis=-1)  

    if isinstance(sigma_in,list)==False:
        # Handle sigma or covariance
        if sigma_in is None:
            W = 66
            sigma = W / (2 * np.sqrt(2 * np.log(2)))
            cov_tmp = [[sigma**2, 0], [0, sigma**2]]
        elif np.isscalar(sigma_in):
            cov_tmp = [[sigma_in**2, 0], [0, sigma_in**2]]
        else:
            cov_tmp = sigma_in
        cov_list = [cov_tmp]*len(len(x))
    elif len(sigma_in)!=len(x): 
        raise "Deve essere almeno un sigma per punto"
        return None 
    else: 
        #ci aspettiamo una lista  di covarianze
        cov_list = sigma_in
    
        

    # Initialize mask
    mask = np.zeros((R, C), dtype=np.float32)

    
    # Add Gaussian for each fixation
    for xi, yi, wi,cov_list_i in zip(x, y, weight_in,cov_list):
        temp_x = int(np.floor(xi))
        temp_y = int(np.floor(yi))
        if np.isscalar(cov_list_i): 
            cov_ = [[cov_list_i**2, 0], [0, cov_list_i**2]]
        else:  
            cov_=cov_list_i
        # Skip se out of bounds
        if temp_x < 0 or temp_x >= C or temp_y < 0 or temp_y >= R:
            continue
        mvn = multivariate_normal(mean=[xi, yi], cov=cov_)
        mask += wi * mvn.pdf(pos)

    # Normalize
    if np.max(mask) > 0:
        mask /= np.max(mask)

    return mask
def make_gauss_mask_for_fixdata(x, y, my_size, fix_time=None,sigma_in=None):
    #Genera fixation map usando una gaussiana centrata nel punto (x,y) di fixation
    
    if fix_time is None:
        fix_time = np.ones_like(x)
        
    if sigma_in is None:
        # Full width at half max in pixels (1 degree of visual angle = 66 pixels)
        #usiamo la relazione tra sigma e FWHM che denota la differenta tra (xi e xi+k) corrispondenti alla 
        #metà del valore max della distribuzione
        W = 1 * 66 #si considera 1°angolo visivo pari a a66 pixel
        sigma = W / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
    else: 
        sigma = sigma_in

    R, C = my_size
    big_R, big_C = 2 * R + 1, 2 * C + 1

    # Create large Gaussian kernel
    X, Y = np.meshgrid(np.arange(big_C), np.arange(big_R))
    temp_num = (X - C) ** 2 + (Y - R) ** 2
    temp = np.exp(-temp_num / (2 * sigma ** 2))
    big_gauss = temp / (2 * np.pi * sigma**2)

    #maschera di output
    my_mask = np.zeros((R, C), dtype=np.float32)

    for i in range(len(x)):
        temp_x = int(np.floor(x[i]))
        temp_y = int(np.floor(y[i]))
        # Skip se out of bounds
        if temp_x < 0 or temp_x >= C or temp_y < 0 or temp_y >= R:
            continue
        # Extract region from big_gauss centered at (temp_x, temp_y)
        gauss_crop = big_gauss[R - temp_y:R - temp_y + R, C - temp_x:C - temp_x + C]
        my_mask += gauss_crop * fix_time[i]  # Weight by fixation time

    # Normalize the mask
    if np.max(my_mask) > 0:
        my_mask  /= np.max(my_mask)

    return my_mask
def export_mask2image(my_mask,path):
    img = (my_mask * 255).astype(np.uint8)
    cv2.imwrite(path, img)
def clamp_rect_to_frame(x, y, w, h, frame_shape):
    #definisci rettangolo in modo tale da stare dentro il frame di input
    frame_height, frame_width = frame_shape[:2]

    x_new = max(0, x)
    y_new = max(0, y)
    w_new = min(frame_width, x + w) - x 
    h_new = min(frame_height, y + h) -y

    return x_new, y_new, w_new, h_new
def get_window_frame_ids(frame_cnt, target_index, window_size=7):
    window_half = window_size // 2
    window_frame_idx_list=[]
    for offset in range(-window_half, window_half + 1):
        idx = target_index + offset
        # setta index a un id valido di frame valido
        idx = max(0, min(idx, frame_cnt - 1))
        window_frame_idx_list.append(idx)
    return window_frame_idx_list
#campioniamo K punti assumendo distribuzione gaussian. nel paper la distribuzione è assunta isotropica quindi 
#possiamo usare difatto euguale std sia a destra che a sinistra della distribuzione assunta 
#Nb. potevamo usare anche altre distribuzioni ma visto che plotteremo come blob gaussiane usiamo questa
def sampling_angles_from_norm(angle,offset,n_sample): 
    #angle-offset = 10th quantile 
    #angle + offset = 90 quantile 
    #poichè z = (x-media)/std 
    #se vogliamo calcolare std =>  std = (x-media)/z 
    #nel nostro caso x = (angle -offset) => 
    # std = ((angle-offset)-angle)/-z = -offset/-z= offset/1.28 dove 1.28 è zscore associato al 10th assumendo distribuzione gaussian
    #corrisponde a quantile 10 (-1.2816) o 90 (1.2816)
    
    
    
    z_score = 1.2816
    std_estimated = offset/z_score
    # Campionamento da distribuzione normale isotropica
    sampled_angles = np.random.normal(loc=angle, scale=std_estimated, size=n_sample).tolist() 
    return sampled_angles 
def spherical2cartesial(x):
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])
    return output
def estimate_2d_gaze_endpoint(origin, gaze_vector, image_shape, scale_factor=1.5,face_size=None,prospective_eq=True):
    #stima 2d del vettore di gaze 
    #qui l'origine è il punto centrale tra occhi o eventualmente centro faccia
    x, y = origin
    gx, gy, gz = gaze_vector
    if gz>=0: 
        return None 
    height, width = image_shape
    if face_size is None:
        face_size=height/4
    k =scale_factor*face_size  
    #print('profondità k=>',k)
    #qui il segno va cambiato perchè ad es. se il soggetto guarda alla sua sx gaze3d.x<0 ma propriettato nell'immagine guarda a destra
    if prospective_eq==False:
        gz = 1
    end_x = x +k * (gx/gz) 
    #se gy>0 soggetto guarda su che significa lato immagine dovremo andare verso lo 0 quindi usare meno anche qui
    end_y = y + k * (gy/gz)
    return (end_x, end_y,k)
def compute_roc_auc(binary_map_true, gaussian_map_estimated):
    #Calcola AUC della curva di roc da usare come score di valutazione. E' fondamentale avere la mappa binaria e quella invece a cui verrnano applicate le soglie
   
    y_true = binary_map_true.flatten()
    y_pred = gaussian_map_estimated.flatten()

    # Compute ROC AUC
    auc_score = roc_auc_score(y_true, y_pred)
    return auc_score
def compute_auc_pr(binary_map_true, gaussian_map_estimated):
    
    #Calcola AUC della curva Precision-Recall tra mappa binaria e mappa continua.
    
    y_true = binary_map_true.flatten()
    y_scores = gaussian_map_estimated.flatten()

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc_score = auc(recall, precision)
    return pr_auc_score
def compute_nss(binary_map_true, gaussian_map_estimated):
    
    #Calcola NSS tra mappa predetta (continua) e mappa binaria dei punti di fissazione.
    
    if np.sum(binary_map_true) == 0:
        return 0.0  # Evita divisione per zero

    pred = gaussian_map_estimated.copy()
    pred = (pred - np.mean(pred)) / (np.std(pred) + 1e-8)  # Normalizza
    nss_score = np.mean(pred[binary_map_true > 0])
    return nss_score
def compute_similarity(map_true, map_est):
    
    #Calcola Similarity tra due mappe continue normalizzate.
    
    true = map_true / (np.sum(map_true) + 1e-8)
    est = map_est / (np.sum(map_est) + 1e-8)
    sim_score = np.sum(np.minimum(true, est))
    return sim_score
#stima dei valori positivi nella mappa target (nb input è un dataset torch)
def estimate_pos_weight(dataset,use_binary = None):
    total_positives = 0
    total_pixels = 0
    used_binary_ = None
    for i in range(len(dataset)):
        _, continuous_target, binary_target = dataset[i]  
        
        if use_binary is not None:
            if use_binary: 
                target_tensor = binary_target 
                used_binary_ = True
            else:
                target_tensor = continuous_target
                used_binary_=False
        #continuous non è sempre presente in base al case (a dfferenza del binary)
        elif isinstance(continuous_target,list) and len(continuous_target)==0:  
            target_tensor=  binary_target
            used_binary_ = True
        else:  
            target_tensor = continuous_target
            used_binary_ = False
        
        #print('target shape',target.shape)
        target_tensor = target_tensor if isinstance(target_tensor, torch.Tensor) else transforms.ToTensor()(target_tensor)
        total_positives += torch.sum(target_tensor > 0).item()
        total_pixels += target_tensor.numel()

    total_negatives = total_pixels - total_positives
    pos_weight = total_negatives / total_positives
    return pos_weight,used_binary_