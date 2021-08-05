# Remote Sensing Change Detection Based on Multidirectional Adaptive Feature Fusion and Perceptual Similarity
by Jialang Xu (e-mail: 504006937@qq.com), Chunbo Luo, Xinyue Chen, Shicai Wei, Yang Luo

PyTorch implementation for "[Remote Sensing Change Detection Based on Multidirectional Adaptive Feature Fusion and Perceptual Similarity](https://www.mdpi.com/2072-4292/13/15/3053)"

- [03 Augest 2021] Release the code of MFPNet model.

## Introduction
Remote sensing change detection (RSCD) is an important yet challenging task in Earth observation. The booming development of convolutional neural networks (CNNs) in computer vision raises new possibilities for RSCD, and many recent RSCD methods have introduced CNNs to achieve promising improvements in performance. This paper proposes a novel multidirectional fusion and perception network for change detection in bi-temporal very-high-resolution remote sensing images. First, we propose an elaborate feature fusion module consisting of a multidirectional fusion pathway (MFP) and an adaptive weighted fusion (AWF) strategy  for RSCD to boost the way that information propagates in the network. The MFP enhances the flexibility and diversity of information paths by creating extra top-down and shortcut-connection paths. The AWF strategy conducts weight recalibration for every fusion node to highlight salient feature maps and overcome semantic gaps between different features. Second, a novel perceptual similarity module is designed to introduce perceptual loss into the RSCD task, which adds the perceptual information, such as structure and semantic, for high-quality change maps generation. Extensive experiments on four challenging benchmark datasets demonstrate the superiority of the proposed network comparing with eight state-of-the-art methods in terms of F1, Kappa, and visual qualities.

## Content
### Architecture
<img src="https://github.com/wzjialang/MFPNet/blob/main/figure/MFPNet.png" height="500"/>

Fig.1 Overall architecture of the proposed multidirectional fusion and perception network (MFPNet). <br>
Note that the process with the dashed line only participates in model training.

### Datasets
The available datasets can be downloaded from the table below:
<table>
	<tr>
	    <th>Datasets</th>
	    <th>Download</th>
	</tr>
    <tr>
	    <td>Season-varying Dataset [<a href="#Ref-1">1</a>]</td>
        <td>[<a href="https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9" target="_blank">Official</a>] </td>
	</tr>
	    <td>LEVIR-CD Dataset [<a href="#Ref-2">2</a>]</td>
        <td>[<a href="https://justchenhao.github.io/LEVIR/" target="_blank">Official</a>]</td>
    </tr>
	</tr>
	    <td>Google Dataset [<a href="#Ref-3">3</a>]</td>
        <td>[<a href="https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset" target="_blank">Official</a>]</td>
    </tr>
	</tr>
	    <td>Zhange Dataset [<a href="#Ref-4">4</a>]</td>
        <td>[<a href="https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery" target="_blank">Official</a>]</td>
    </tr>
</table> 

### Requirements
- Python 3.6
- PyTorch 1.1.0
- cudatoolkit 9.0
- cudnn 7.6.5
- OpenCV-Python 3.4.2

## Reference
<span id="Ref-1">[1] Lebedev, M.; Vizilter, Y.V.; Vygolov, O.; Knyaz, V.; Rubis, A.Y. Change Detection in Remote Sensing Images Using Conditional Adversarial Networks. Int. Archives Photogram. Remote Sens. Spatial Inf. Sci. 2018, 42, 565–571.</span>

<span id="Ref-2">[2] Chen, H.; Shi, Z. A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection. Remote Sens. 2020, 12, 1662. </span>

<span id="Ref-3">[3] Peng, D.; Bruzzone, L.; Zhang, Y.; Guan, H.; Ding, H.; Huang, X. SemiCDNet: A Semisupervised Convolutional Neural Network for Change Detection in High Resolution Remote-Sensing Images. IEEE Trans. Geosci. Remote Sens. 2020, pp. 1–16. </span>

<span id="Ref-4">[4] Zhang, C.; Yue, P.; Tapete, D.; Jiang, L.; Shangguan, B.; Huang, L.; Liu, G. A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images. ISPRS-J. Photogramm. Remote Sens. 2020, 166, 183–200. </span>

## Acknowledgement
Thanks [zylo117](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) for his great work on 'Conv2dStaticSamePadding' and 'MaxPool2dStaticSamePadding'!

## Cite
```
@Article{rs13153053,
AUTHOR = {Xu, Jialang and Luo, Chunbo and Chen, Xinyue and Wei, Shicai and Luo, Yang},
TITLE = {Remote Sensing Change Detection Based on Multidirectional Adaptive Feature Fusion and Perceptual Similarity},
JOURNAL = {Remote Sensing},
VOLUME = {13},
YEAR = {2021},
NUMBER = {15},
ARTICLE-NUMBER = {3053},
URL = {https://www.mdpi.com/2072-4292/13/15/3053},
ISSN = {2072-4292},
DOI = {10.3390/rs13153053}
}
```
