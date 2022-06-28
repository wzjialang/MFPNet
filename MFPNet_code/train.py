import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics)
from sklearn.metrics import precision_recall_fscore_support as prfs
import os
import logging
import json
import random
import numpy as np
import re
import warnings
from models.vgg import Vgg19
warnings.filterwarnings("ignore")

"""
Initialize Parser and define arguments
"""
parser, metadata = get_parser_with_args(metadata_json_path='/home/aaa/xujialang/master_thesis/MFPNet/metadata.json')
opt = parser.parse_args()

"""
Initialize experiments log
"""
logging.basicConfig(level=logging.INFO)

"""
Set up environment: define paths, download data, and set device
"""
dev = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(seed=777)

train_loader, val_loader = get_loaders(opt)
print(opt.batch_size * len(train_loader))
print(opt.batch_size * len(val_loader))

"""
Load Model then define other aspects of the model
"""
logging.info('LOADING Model')
model = load_model(opt, dev)
vgg=Vgg19().to(dev)
"""
 Resume
"""
epoch_resume=0
if opt.resume != "None":
    model.load_state_dict(torch.load(os.path.join(opt.resume)))
    epoch_resume=int(re.sub("\D","",opt.resume))
    print('resume success: epoch {}'.format(epoch_resume))

criterion_ce = nn.CrossEntropyLoss().to(dev)
criterion_perceptual = nn.MSELoss().to(dev)
criterion = get_criterion(opt)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=2, eta_min=0, last_epoch=-1)

"""
 Set starting values
"""
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
logging.info('STARTING training')

for epoch in range(opt.epochs):
    epoch= epoch + epoch_resume +1
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()
    
    """
    Begin Training
    """
    model.train()
    logging.info('SET model mode to train!')

    for batch_img1, batch_img2, labels in train_loader:
        # Set variables for training
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        # Zero the gradient
        optimizer.zero_grad()

        # Get model predictions, calculate loss, backprop
        cd_preds= model(batch_img1, batch_img2)
        loss = criterion(criterion_ce, criterion_perceptual, cd_preds, labels, batch_img1, vgg, dev)
        
        loss.backward()
        optimizer.step()
    
        # Calculate and log other batch metrics
        cd_preds = torch.argmax(cd_preds, dim = 1)
        cd_corrects = (100 *
                       (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                       (labels.size()[0] * (opt.patch_size**2)))
        cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
                               cd_preds.data.cpu().numpy().flatten(),
                               average='binary',
                               pos_label=1)
        train_metrics = set_metrics(train_metrics,
                                    loss,
                                    cd_corrects,
                                    cd_train_report,
                                    scheduler.get_last_lr())
        
        # log the batch mean metrics
        mean_train_metrics = get_mean_metrics(train_metrics)
        
        # clear batch variables from memory
        del batch_img1, batch_img2, labels
    
    scheduler.step()
    logging.info("EPOCH {} TRAIN METRICS. ".format(epoch) + str(mean_train_metrics))


    """
    Begin Validation
    """
    model.eval()
    with torch.no_grad():
        for batch_img1, batch_img2, labels in val_loader:
            # Set variables for training
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            # Get predictions and calculate loss
            cd_preds = model(batch_img1, batch_img2)
            val_loss = criterion(criterion_ce, criterion_perceptual, cd_preds, labels, batch_img1, vgg, dev)

            # Calculate and log other batch metrics
            cd_preds = torch.argmax(cd_preds, dim = 1)
            cd_corrects = (100 *
                           (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                           (labels.size()[0] * (opt.patch_size**2)))
            cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                                 cd_preds.data.cpu().numpy().flatten(),
                                 average='binary',
                                 pos_label=1)
            val_metrics = set_metrics(val_metrics,
                                      val_loss,
                                      cd_corrects,
                                      cd_val_report,
                                      scheduler.get_lr())

            # log the batch mean metrics
            mean_val_metrics = get_mean_metrics(val_metrics)

            # clear batch variables from memory
            del batch_img1, batch_img2, labels

        logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(mean_val_metrics))
   
        """
        Store the weights of good epochs based on validation results
        """
        if (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores']):
            # Insert training and epoch information to metadata dictionary
            logging.info('updata the model')
            metadata['val_metrics'] = mean_val_metrics

            # Save model and log
            if not os.path.exists(opt.weight_dir):
                os.mkdir(opt.weight_dir)
            with open(opt.weight_dir + 'metadata_val_epoch_' + str(epoch) + '.json', 'w') as fout:
                json.dump(metadata, fout)

            torch.save(model.state_dict(), opt.weight_dir + 'checkpoint_epoch_'+str(epoch)+'_f1_'+str(mean_val_metrics['cd_f1scores'])+'.pt')
            best_metrics = mean_val_metrics
            print('best val: ' + str(mean_val_metrics))

    print('An epoch finished.')

print('Done!')