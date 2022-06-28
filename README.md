# Remote Sensing Change Detection Based on Multidirectional Adaptive Feature Fusion and Perceptual Similarity
by Jialang Xu (e-mail: 504006937@qq.com), Chunbo Luo, Xinyue Chen, Shicai Wei, Yang Luo

PyTorch implementation for "[Remote Sensing Change Detection Based on Multidirectional Adaptive Feature Fusion and Perceptual Similarity](https://www.mdpi.com/2072-4292/13/15/3053)"

- [03 August 2021] Release the code of MFPNet model.
- [Coming soon] Release the processed datasets and the training/evaluation code.

## Introduction
Remote sensing change detection (RSCD) is an important yet challenging task in Earth observation. The booming development of convolutional neural networks (CNNs) in computer vision raises new possibilities for RSCD, and many recent RSCD methods have introduced CNNs to achieve promising improvements in performance. This paper proposes a novel multidirectional fusion and perception network for change detection in bi-temporal very-high-resolution remote sensing images. First, we propose an elaborate feature fusion module consisting of a multidirectional fusion pathway (MFP) and an adaptive weighted fusion (AWF) strategy  for RSCD to boost the way that information propagates in the network. The MFP enhances the flexibility and diversity of information paths by creating extra top-down and shortcut-connection paths. The AWF strategy conducts weight recalibration for every fusion node to highlight salient feature maps and overcome semantic gaps between different features. Second, a novel perceptual similarity module is designed to introduce perceptual loss into the RSCD task, which adds the perceptual information, such as structure and semantic, for high-quality change maps generation. Extensive experiments on four challenging benchmark datasets demonstrate the superiority of the proposed network comparing with eight state-of-the-art methods in terms of F1, Kappa, and visual qualities.

## Content
### Architecture
<img src="https://github.com/wzjialang/MFPNet/blob/main/figure/MFPNet.png" height="500"/>

Fig.1 Overall architecture of the proposed multidirectional fusion and perception network (MFPNet). <br>
Note that the process with the dashed line only participates in model training.

### Datasets
The processed and original datasets can be downloaded from the table below, we recommended downloading the processed one directly to get a quick start on our code:

<table>
	<tr>
	    <th>Datasets</th>
	    <th>Processed Links</th>
	    <th>Original Links</th>	
	</tr>
    <tr>
	    <td>Season-varying Dataset [<a href="https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-2/565/2018/">1</a>]</td>
	    <td rowspan="4">[<a href="" target="_blank">Google Drive</a>]
	    [<a href="https://pan.baidu.com/s/1kf5QmTY8Usnknkao1JcAkw?pwd=1234" target="_blank">Baidu Drive</a>] 
        <td>[<a href="https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9" target="_blank">Original</a>] </td>
	</tr>
	    <td>LEVIR-CD Dataset [<a href="https://www.mdpi.com/2072-4292/12/10/1662">2</a>]</td>
	    <td>[<a href="https://justchenhao.github.io/LEVIR/" target="_blank">Original</a>]</td>
    </tr>
	</tr>
	    <td>Google Dataset [<a href="https://ieeexplore.ieee.org/document/9161009/">3</a>]</td>
	    <td>[<a href="https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset" target="_blank">Original</a>]</td>
    </tr>
	</tr>
	    <td>Zhange Dataset [<a href="https://www.sciencedirect.com/science/article/abs/pii/S0924271620301532">4</a>]</td>
	    <td>[<a href="https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery" target="_blank">Original</a>]</td>
    </tr>
</table> 

### Setup & Usage for the Code

1. Check the structure of data folders:
```
(root folder)
├── dataset1
|  ├── train
|  |  ├── A
|  |  ├── B
|  |  ├── label
|  ├── val
|  |  ├── A
|  |  ├── B
|  |  ├── label
|  ├── test
|  |  ├── A
|  |  ├── B
|  |  ├── label
├── ...
```

2. Check dependencies:
```
- Python 3.6+
- PyTorch 1.7.0+
- scikit-learn
- cudatoolkit
- cudnn
- OpenCV-Python
```

3. Change paths:
```
- Change the 'metadata_json_path' in 'train.py' to your 'metadata.json' path.
- Change the 'dataset_dir' and 'weight_dir' in 'metadata.json' to your own path.
```

4. Train the MFPNet:
```
python train.py
```

5. Evaluate the MFPNet:
```
- Change the 'weight_path' in 'eval.py' to your model weight path.
- python eval.py
```

## Reference
Appreciate the work from the following repositories:
* [likyoo/Siam-NestedUNet](https://github.com/likyoo/Siam-NestedUNet)
* [zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

## Cite
If this repository is useful for your research, please cite:
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
