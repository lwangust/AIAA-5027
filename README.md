# Deep Learning for Visual Intelligence: Trends and Challenges 

## Course information

* Instructor: WANG Lin (linwang@ust.hk)
* TAs: SU Ying (ysuay@connect.ust.hk) and ZHU Qingyan (qzhuai@connect.ust.hk)
* Office Hours: BY appointment only.

## Grading policy 
- Paper summary (10%)
- Paper presentation and discussion (30%)
- Group project and paper submission (50%)
- Attendance and Participation (10%)

##  Tentative schedule
| Dates | Topics | Active Learning |
| --- | --- | --- |
| 2/8 | Course introduction | |
| 2/10 | Course introduction | Overview of computer vision |
| 2/15 | Deep learning basics | TAs’ lectures for CNN basics, algorithm basics and Pytorch tuorial  |
| 2/17 | Deep learning basics | TAs’ lectures for CNN basics, algorithm basics and Pytorch tuorial  |
| 2/22 | DNN models in computer vision (GAN, RNN, GNN) |   |
| 2/24 | DNN models in computer vision (GAN, RNN, GNN) |  (1) Persenation (2) Review due 2/27 (3) Project meetings |
| 3/1 | Learning methods in computer vision (Transfer learning, domain adaptation, self/semi-supervised learning) |   |
| 3/3 | Learning methods in computer vision ((Transfer learning, domain adaptation, self/semi-supervised learning)) |  (1) Persenation (2) Review due 3/6  |
| 3/8 |Deep learning for image restoration and enhancement (I) deblurring, deraining, dehazing |   |
| 3/10 |Deep learning for image restoration and enhancement (I) deblurring, deraining, dehazing  |  (1) Persenation (2) Review due 3/13 (3) Project proposal kick-off (one page) |
| 3/15 |Deep learning for image restoration and enhancement (II) Super-resolution, HDR imaging |   |
| 3/17 |Deep learning for image restoration and enhancement (II) Super-resolution, HDR imaging  |  (1) Persenation (2) Review due 3/20 |
| 3/22 |Deep learning for scene understanding (I) Object detection & tracking |  |
| 3/24 |Deep learning for scene understanding (I) Object detection & tracking | Project mid-term presentation |
| 3/29 |Deep learning for scene understanding (II) Semantic segmentation  |  |
| 3/31 |Deep learning for scene understanding (II) Semantic segmentation  | (1) Persenation (2) Review due 4/3 |
| 4/5 |Computer vision with novel cameras (I) Event camera-based vision  |  |
| 4/7 |Computer vision with novel cameras (I) Event camera-based vision  | (1) Persenation (2) Review due 4/10 |
| 4/12 |Computer vision with novel cameras (II) Thermal/360 camera-based vision  |  |
| 4/14 |Computer vision with novel cameras (II) Thermal/360 camera-based vision  | (1) Persenation (2) Review due 4/17 (3) Project meetings |
| 4/19 |Special vision problems (Learning  in adverse visual conditions) |  |
| 4/21 |Special vision problems (Learning  in adverse visual conditions) | (1) Persenation (2) Review due 4/24 |
| 4/26 |Adversarial robustness in computer vision (Adversrial attack and defense) |  |
| 4/28 |Adversarial robustness in computer vision (Adversrial attack and defense)| (1) Persenation (2) Review due 4/31 (3) Project meetings |
| 5/3 |Potential and Challenges in computer vision (data, computation, learning, sensor) (self-driving and robotics) |  |
| 5/5 |Potential and Challenges in computer vision (data, computation, learning, sensor) (self-driving and robotics)| (1) TA/Student lectures (2) final project Q/A  |
| 5/10 |Project presentation and final paper submission |  |
| 5/12 |Project presentation and final paper submission | Submission due 5/26  |


##  Reading list

### DNN models in computer vision (VAEs, GANs, RNNs)
#### VAEs
[[Kingma and Welling 14]](https://arxiv.org/pdf/1312.6114v10.pdf) Auto-Encoding Variational Bayes, ICLR 2014. </br>
[[Kingma et al. 15]](https://arxiv.org/pdf/1506.02557.pdf) Variational Dropout and the Local Reparameterization Trick, NIPS 2015.</br>
[[Blundell et al. 15]](https://arxiv.org/pdf/1505.05424.pdf) Weight Uncertainty in Neural Networks, ICML 2015.</br>
[[Gal and Ghahramani 16]](http://proceedings.mlr.press/v48/gal16.pdf) Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning, ICML 2016. </br>

#### GANs
[[Goodfellow et al. 14] ](https://arxiv.org/pdf/1406.2661.pdf)Generative Adversarial Nets, NIPS 2014. </br>
[[Radford et al. 15] ](https://arxiv.org/pdf/1809.11096.pdf)Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, ICLR 2016. </br>
[[Chen et al. 16]](https://arxiv.org/pdf/1606.03657.pdf) InfoGAN: Interpreting Representation Learning by Information Maximizing Generative Adversarial Nets, NIPS 2016. </br>
[[Arjovsky et al. 17]](http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf) Wasserstein Generative Adversarial Networks, ICML 2017. </br>
[[Zhu et al. 17]](https://arxiv.org/pdf/1703.10593.pdf) Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, ICCV 2017.</br>
[[Liu et al. 17]](https://arxiv.org/pdf/1703.00848.pdf) UNIT: Unsupervised Image-to-Image Translation Networks, NeurIPS 2017. </br>
[[Choi et al. 18]](https://arxiv.org/pdf/1711.09020.pdf)StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation, CVPR 2018.  </br>
[[Brock et al. 19]](https://arxiv.org/pdf/1809.11096.pdf) Large Scale GAN Training for High-Fidelity Natural Image Synthesis, ICLR 2019. </br>
[[Karras et al. 19] ](https://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf) A Style-Based Generator Architecture for Generative Adversarial Networks, CVPR 2019. </br>
[[Karras et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.pdf) Analyzing and Improving the Image Quality of StyleGAN, CVPR 2020. </br>
[[Park et al. 20] ](https://arxiv.org/pdf/2007.15651.pdf) Contrastive Learning for Unpaired Image-to-Image Translation, ECCV 2020. </br> 
[[Karras et al. 20]](https://arxiv.org/pdf/2006.06676.pdf) Training Generative Adversarial Networks with Limited Data, NeurIPS 2020. </br> 
[[Xie et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650494.pdf) Self-Supervised CycleGAN for Object-Preserving Image-to-Image Domain Adaptation, ECCV 2020. </br>
[[Mustafa et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630579.pdf) Transformation Consistency Regularization– A Semi-Supervised Paradigm for
Image-to-Image Translation, ECCV 2020. </br>
[[Li et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710647.pdf) Semantic Relation Preserving Knowledge Distillation for Image-to-Image Translation, ECCV, 2020.  </br>
[[Xu et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Linear_Semantics_in_Generative_Adversarial_Networks_CVPR_2021_paper.pdf) Linear Semantics in Generative Adversarial Networks, CVPR, 2021.  </br>
[[Cao et al. 21]](https://arxiv.org/pdf/2103.16835.pdf) ReMix: Towards Image-to-Image Translation with Limited Data, CVPR 2021.  </br>
[[Liu et al. 21]](https://arxiv.org/pdf/2103.07893.pdf) DivCo: Diverse Conditional Image Synthesis via Contrastive Generative Adversarial Network, CVPR 2021.  </br>
[[Pizzati et al. 21]](https://arxiv.org/pdf/2103.06879.pdf) CoMoGAN: continuous model-guided image-to-image translation, CVPR 2021.  </br>
[[Jin et al. 21]](https://arxiv.org/pdf/2103.03467.pdf) Teachers Do More Than Teach: Compressing Image-to-Image Models, CVPR 2021.   </br>
[[Baek et al. 21]](https://arxiv.org/pdf/2006.06500.pdf) Rethinking the Truly Unsupervised Image-to-Image Translation, ICCV, 2021.   </br>
[[Wang et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_TransferI2I_Transfer_Learning_for_Image-to-Image_Translation_From_Small_Datasets_ICCV_2021_paper.pdf) TransferI2I: Transfer Learning for Image-to-Image Translation from Small Datasets, ICCV, 2021. </br>
[[Yang et al. 21]](https://arxiv.org/pdf/2111.10346.pdf) Global and Local Alignment Networks for Unpaired Image-to-Image Translation, Arxiv 2021. 



