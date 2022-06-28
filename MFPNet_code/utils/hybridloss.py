import torch
import torch.nn.functional as F

def hybrid_loss(criterion_ce, criterion_perceptual, prediction, target, img_prev_train, vgg, dev):
    """Calculating the loss"""
    loss = 0
    
    # Perceptual Similarity Module (PSM) 
    out_train_softmax2d = F.softmax(prediction,dim=1)
    an_change = out_train_softmax2d[:,1,:,:].unsqueeze(1).expand_as(img_prev_train)
    an_unchange = out_train_softmax2d[:,0,:,:].unsqueeze(1).expand_as(img_prev_train)
    label_change = target.unsqueeze(1).expand_as(img_prev_train).type(torch.FloatTensor).to(dev)
    label_unchange = 1-label_change
    an_change = an_change * label_change
    an_unchange = an_unchange * label_unchange

    an_change_feature = vgg(an_change)
    gt_feature = vgg(label_change)   
    an_unchange_feature = vgg(an_unchange)
    gt_feature_unchange = vgg(label_unchange)
    
    perceptual_loss_change = criterion_perceptual(an_change_feature[0], gt_feature[0])
    perceptual_loss_unchange = criterion_perceptual(an_unchange_feature[0], gt_feature_unchange[0])
    perceptual_loss = perceptual_loss_change + perceptual_loss_unchange

    loss = 0.0001*perceptual_loss + criterion_ce(prediction, target)

    return loss

