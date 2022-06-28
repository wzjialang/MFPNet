from shutil import copyfile
import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn.functional as F
import cv2
import os
from utils.helpers import load_model

parser, metadata = get_parser_with_args(metadata_json_path='/home/aaa/xujialang/master_thesis/MFPNet/metadata.json')
opt = parser.parse_args()
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt)

weight_path = os.path.join(opt.weight_dir, 'model_weight.pt')   # the path of the model weight
model = load_model(opt, dev)
model.load_state_dict(torch.load(weight_path))
"""
Begin Test
"""
model.eval()
with torch.no_grad():
    c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    test_metrics = {
        'cd_precisions': [],
        'cd_recalls': [],
        'cd_f1scores': [],
        }

    for batch_img1, batch_img2, labels in test_loader:
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)
        cd_preds = model(batch_img1, batch_img2)
        cd_preds = torch.argmax(cd_preds, dim = 1)

        tp= (labels.cpu().numpy() * cd_preds.cpu().numpy()).sum()
        tn= ((1-labels.cpu().numpy()) * (1-cd_preds.cpu().numpy())).sum()
        fn= (labels.cpu().numpy() * (1-cd_preds.cpu().numpy())).sum()
        fp= ((1-labels.cpu().numpy()) * cd_preds.cpu().numpy()).sum()
        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp

    tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * P * R / (R + P)
    IOU = tp/ (fn+tp+fp)
    
    ttt_test=tn+fp+fn+tp
    TA_test = (tp+tn) / ttt_test
    Pcp1_test = (tp + fn) / ttt_test
    Pcp2_test = (tp + fp) / ttt_test
    Pcn1_test = (fp + tn) / ttt_test
    Pcn2_test = (fn + tn) / ttt_test
    Pc_test = Pcp1_test*Pcp2_test + Pcn1_test*Pcn2_test
    kappa_test = (TA_test - Pc_test) / (1 - Pc_test)

    test_metrics['cd_f1scores'] = F1
    test_metrics['cd_precisions'] = P
    test_metrics['cd_recalls'] = R
    print("TEST METRICS. KAPPA: {}. IOU: {} ".format(kappa_test, IOU) + str(test_metrics))