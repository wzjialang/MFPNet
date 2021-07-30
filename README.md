# Remote Sensing Change Detection Based on Multidirectional Adaptive Feature Fusion and Perceptual Similarity
by Jialang Xu (e-mail: 504006937@qq.com), Chunbo Luo, Xinyue Chen, Shicai Wei, Yang Luo

PyTorch implementation for "Remote Sensing Change Detection Based on Multidirectional Adaptive Feature Fusion and Perceptual Similarity"

- I will release the code when the paper is published.

## Introduction
Remote sensing change detection (RSCD) is an important yet challenging task in Earth observation. The booming development of convolutional neural networks (CNNs) in computer vision raises new possibilities for RSCD, and many recent RSCD methods have introduced CNNs to achieve promising improvements in performance. This paper proposes a novel multidirectional fusion and perception network for change detection in bi-temporal very-high-resolution remote sensing images. First, we propose an elaborate feature fusion module consisting of a multidirectional fusion pathway (MFP) and an adaptive weighted fusion (AWF) strategy  for RSCD to boost the way that information propagates in the network. The MFP enhances the flexibility and diversity of information paths by creating extra top-down and shortcut-connection paths. The AWF strategy conducts weight recalibration for every fusion node to highlight salient feature maps and overcome semantic gaps between different features. Second, a novel perceptual similarity module is designed to introduce perceptual loss into the RSCD task, which adds the perceptual information, such as structure and semantic, for high-quality change maps generation. Extensive experiments on four challenging benchmark datasets demonstrate the superiority of the proposed network comparing with eight state-of-the-art methods in terms of F1, Kappa, and visual qualities.

## Content
### Datasets
The available datasets can be downloaded from the table below:
<table>
	<tr>
	    <th>Datasets</th>
	    <th>Download</th>
	</tr>
    <tr>
	    <td>Season-varying Dataset</td>
        <td>[<a href="https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9" target="_blank">Official</a>] </td>
	</tr>
	    <td>LEVIR-CD Dataset</td>
        <td>[<a href="https://justchenhao.github.io/LEVIR/" target="_blank">Official</a>]</td>
    </tr>
	</tr>
	    <td>Google Dataset</td>
        <td>[<a href="https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset" target="_blank">Official</a>]</td>
    </tr>
	</tr>
	    <td>Zhange Dataset</td>
        <td>[<a href="https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery" target="_blank">Official</a>]</td>
    </tr>
</table> 

### Requirements
- Python 3.6
- PyTorch 1.1.0
- cudatoolkit 9.0
- cudnn 7.6.5
- OpenCV-Python 3.4.2
