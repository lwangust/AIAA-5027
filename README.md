# Deep Learning for Visual Intelligence: Trends and Challenges 

## Course information

* Instructor: WANG Lin (linwang@ust.hk)
* TAs: SU Ying (ysuay@connect.ust.hk) and ZHU Qingyan (qzhuai@connect.ust.hk)
* Class time: Tuesday & Thursday 16.30 -17.50 
* Office Hours: BY appointment only.

## Course description

This is a task-oriented yet interaction-based course, which aims to scrutinize the recent trends and challenges of deep learning in visual intelligence tasks (learning methods, high- and low-level vision problems). This course will follow the way of flipped-classroom manner where the lecturer teaches the basics; meanwhile, the students will also be focused on active discussions, presentations (lecturing), and hands-on research projects under the guidance of the lecturer in the whole semester. Through this course, students will be equipped with the capability to critically challenge the existing methodologies/techniques and hopefully make breakthroughs in some new research directions.

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
[[Yang et al. 21]](https://arxiv.org/pdf/2111.10346.pdf) Global and Local Alignment Networks for Unpaired Image-to-Image Translation, Arxiv 2021. </br>
[[Jiang et al. 21]](https://arxiv.org/pdf/2012.12821v3.pdf)  Focal Frequency Loss for Image Reconstruction and Synthesis, ICCV, 2021.

### Learning methods in computer vision
#### Knowledge transfer 
[[Wang et al. 21]](https://arxiv.org/pdf/2004.05937v7.pdf) Knowledge Distillation and Student-Teacher Learning for Visual Intelligence: A Review and New Outlooks, TPAMI, 2021. </br>
[[Hiton et al. 15]](https://arxiv.org/pdf/1503.02531.pdf) Distilling the Knowledge in a Neural Network, NIPS Workshop, 2015. </br>
[[Romero et al. 15]](https://arxiv.org/pdf/1412.6550.pdf) FitNets: Hints for Thin Deep Nets, ICLR, 2015. </br>
[[Gupta et al. 16]](https://arxiv.org/pdf/1507.00448.pdf) Cross Modal Distillation for Supervision Transfer, CVPR, 2016.    </br>
[[Zagoruyko et al. 16]](https://arxiv.org/pdf/1612.03928.pdf)  Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer, ICLR, 2017.  </br>
[[Furlanello et al. 18]](https://arxiv.org/abs/1805.04770) Born Again Neural Networks, ICML, 2018.   </br>
[[Zhang et al. 18]](https://arxiv.org/pdf/1706.00384.pdf) Deep Mutual Learning, CVPR,2018.  </br>
[[Tarvainen et al. 18]](https://arxiv.org/pdf/1703.01780.pdf)Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results, NIPS, 2018.  </br>
[[Zhang et al. 19]](https://arxiv.org/pdf/1905.08094.pdf)  Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation, ICCV, 2019.  </br> 
[[Heo et al. 19]](https://arxiv.org/pdf/1904.01866.pdf) A Comprehensive Overhaul of Feature Distillation, ICCV, 2019.  </br>
[[Tung et al.19]](https://arxiv.org/pdf/1907.09682.pdf) Similarity-Preserving Knowledge Distillation, ICCV, 2019.   </br>
[[Chen et al. 19]](https://www.wangyunhe.site/data/2019%20ICCV%20DAFL.pdf) DAFL:Data-Free Learning of Student Networks, ICCV, 2019. </br>
[[Ahn et al. 19]](https://arxiv.org/pdf/1904.05835.pdf) Variational Information Distillation for Knowledge Transfer, CVPR, 2019.  </br>
[[Tian et al. 20]](https://arxiv.org/pdf/1910.10699v2.pdf)  Contrastive Representation Distillation, ICLR, 2020.  </br>
[[Fang et al. 20]](https://arxiv.org/pdf/1912.11006.pdf) Data-Free Adversarial Distillation, CVPR, 2020.  </br>
[[Yang et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460290.pdf) MutualNet: Adaptive ConvNet via Mutual Learning from Network Width and Resolution, ECCV, 2020. </br>
[[Yao et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600290.pdf) Knowledge Transfer via Dense Cross-layer Mutual-distillation. ECCV 2020 </br>
[[Guo et al. 20]](https://arxiv.org/pdf/2010.07485.pdf) Reducing the Teacher-Student Gap via Spherical Knowledge Disitllation, Arxiv, 2020.  </br>
[[Ji et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ji_Refine_Myself_by_Teaching_Myself_Feature_Refinement_via_Self-Knowledge_Distillation_CVPR_2021_paper.pdf) Refine Myself by Teaching Myself: Feature Refinement via Self-Knowledge Distillation, CVPR, 2021. </br>
[[Liu et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Source-Free_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2021_paper.pdf) Source-Free Domain Adaptation for Semantic Segmentation, CVPR, 2021. </br>
[[Chen et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Learning_Student_Networks_in_the_Wild_CVPR_2021_paper.pdf) Learning Student Networks in the Wild, CVPR, 2021. </br>
[[Xue et a. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xue_Multimodal_Knowledge_Expansion_ICCV_2021_paper.pdf) Multimodal Knowledge Expansion，ICCV, 2021. </br>
[[ZHu et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhu_Student_Customized_Knowledge_Distillation_Bridging_the_Gap_Between_Student_and_ICCV_2021_paper.pdf) Student Customized Knowledge Distillation: Bridging the Gap Between Student and Teacher, ICCV,  2021. </br>
[[Kim et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Self-Knowledge_Distillation_With_Progressive_Refinement_of_Targets_ICCV_2021_paper.pdf) Self-Knowledge Distillation with Progressive Refinement of Targets, ICCV, 2021.   </br>
[[Son et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Son_Densely_Guided_Knowledge_Distillation_Using_Multiple_Teacher_Assistants_ICCV_2021_paper.pdf)  Densely Guided Knowledge Distillation using Multiple Teacher Assistants, ICCV, 2021.  </br>

#### Domain Adaptation
[[Long et al. 15] ](https://arxiv.org/pdf/1502.02791.pdf)Learning Transferable Features with Deep Adaptation Networks, ICML, 2015. </br>
[[Tzeng et al. 17]](https://arxiv.org/pdf/1702.05464.pdf) Adversarial Discriminative Domain Adaptation, CVPR, 2017. </br>
[[Huang et al. 18]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Haoshuo_Huang_Domain_transfer_through_ECCV_2018_paper.pdf) Domain Transfer Through Deep Activation Matching, ECCV, 2018. </br>
[[Bermu’dez-Chaco’n et al. 20]](https://openreview.net/pdf?id=rJxycxHKDS) Domain Adaptive Multibranch Networks, ICLR, 2020. </br>
[[Carlucci et al. 17]](https://arxiv.org/pdf/1704.08082.pdf) AutoDIAL: Automatic DomaIn Alignment Layers, ICCV, 2017.  </br>
[[Chang et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_Domain-Specific_Batch_Normalization_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)  Domain-Specific Batch Normalization for Unsupervised Domain Adaptation, CVPR, 2019. </br>
[[Cui et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cui_Towards_Discriminability_and_Diversity_Batch_Nuclear-Norm_Maximization_Under_Label_Insufficient_CVPR_2020_paper.pdf) Towards Discriminability and Diversity:Batch Nuclear-norm Maximization under Label Insufficient Situations, CVPR 2020. </br>
[[Roy et al. 19]](https://arxiv.org/pdf/1903.03215.pdf) Unsupervised Domain Adaptation using Feature-Whitening and Consensus Loss, CVPR, 2019. </br>
[[Csurka et al. 17]](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w38/Csurka_Discrepancy-Based_Networks_for_ICCV_2017_paper.pdf)  Discrepancy-based networks for unsupervised domain adaptation: a comparative study, CVPRW, 2017. </br>
[[Murez et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Murez_Image_to_Image_CVPR_2018_paper.pdf) Image to Image Translation for Domain Adaptation, CVPR, 2018. </br>
[[Liu et al. 17]](https://arxiv.org/pdf/1606.07536.pdf) Coupled Generative Adversarial Networks, NIPS, 2017. </br>
[[Hoffman et al. 18]](https://arxiv.org/pdf/1711.03213.pdf) CyCADA: Cycle-Consistent Adversarial Domain Adaptation, ICLR, 2018. </br>
[[Lee et al. 18]](https://arxiv.org/pdf/1808.00948.pdf) Diverse Image-to-Image Translation via Disentangled Representations, ECCV, 2018. </br>
[[Chen et al. 12]](https://arxiv.org/ftp/arxiv/papers/1206/1206.4683.pdf) Marginalized Denoising Autoencoders for Domain Adaptation, ICML, 2012. </br>
[[Zhuang et al. 15]](https://www.ijcai.org/Proceedings/15/Papers/578.pdf) Supervised Representation Learning: Transfer Learning with Deep Autoencoders, IJCAI, 2015. </br>
[[ Ghifary et al. 16]](https://arxiv.org/pdf/1607.03516.pdf) Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation, ECCV, 2016.  </br>
[[Bousmalis et al. 16]](https://arxiv.org/pdf/1608.06019.pdf) Domain Separation Networks, NIPS, 2016.  </br>
[[French et al. 19]](https://arxiv.org/pdf/1706.05208.pdf) Self-ensembling for Visual Domain Adaptation, ICLR, 2019.  </br>
[[Shu et al. 18]](https://arxiv.org/pdf/1802.08735.pdf) A DIRT-T Approach to Unsupervised Domain Adaptation, ICLR, 2018. </br>
[[ Deng et al. 19]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Deng_Cluster_Alignment_With_a_Teacher_for_Unsupervised_Domain_Adaptation_ICCV_2019_paper.pdf) Cluster Alignment with a Teacher for Unsupervised Domain Adaptation, ICCV, 2019. </br>
[[Chen et al. 19]](https://arxiv.org/pdf/1811.08585.pdf) Progressive Feature Alignment for Unsupervised Domain Adaptation, CVPR 2019. </br>
[[Zhang et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1410.pdf)  Progressive Feature Alignment for Unsupervised Domain Adaptation, CVPR 2018. </br>
[[Kang et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kang_Contrastive_Adaptation_Network_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)  Contrastive Adaptation Network for Unsupervised Domain Adaptation, CVPR 2019. </br>

---
[[Guizilini et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Guizilini_Geometric_Unsupervised_Domain_Adaptation_for_Semantic_Segmentation_ICCV_2021_paper.pdf)  Geometric Unsupervised Domain Adaptation for Semantic Segmentation, ICCV, 2021. </br>
[[Wang et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530715.pdf)  Learning to Combine: Knowledge Aggregation for Multi-Source Domain Adaptation, ECCV, 2020.  </br>
[[Peng et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510749.pdf) Domain2Vec: Domain Embedding for Unsupervised Domain Adaptation, ECCV, 2020. </br>
[[Liu et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Source-Free_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2021_paper.pdf)  Source-Free Domain Adaptation for Semantic Segmentation, CVPR, 2021. </br>
[[Na et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Na_FixBi_Bridging_Domain_Spaces_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.pdf) FixBi: Bridging Domain Spaces for Unsupervised Domain Adaptation, CVPR, 2021.  </br>
[[Sharma et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Sharma_Instance_Level_Affinity-Based_Transfer_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.pdf) Instance Level Affinity-Based Transfer for Unsupervised Domain Adaptation, CVPR, 2021.  </br>
[[Ahmed et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ahmed_Unsupervised_Multi-Source_Domain_Adaptation_Without_Access_to_Source_Data_CVPR_2021_paper.pdf) 
Unsupervised Multi-source Domain Adaptation Without Access to Source Data, CVPR, 2021. </br>
[[He et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/He_Multi-Source_Domain_Adaptation_With_Collaborative_Learning_for_Semantic_Segmentation_CVPR_2021_paper.pdf) Multi-Source Domain Adaptation with Collaborative Learning for Semantic Segmentation, CVPR, 2021. </br>
[[Wu et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_DANNet_A_One-Stage_Domain_Adaptation_Network_for_Unsupervised_Nighttime_Semantic_CVPR_2021_paper.pdf) DANNet: A One-Stage Domain Adaptation Network for Unsupervised Nighttime Semantic Segmentation, CVPR, 2021. </br>
[[Lengyel et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Lengyel_Zero-Shot_Day-Night_Domain_Adaptation_With_a_Physics_Prior_ICCV_2021_paper.pdf) Zero-Shot Day-Night Domain Adaptation with a Physics Prior, ICCV, 2021. </br>
[[Li et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Semantic_Concentration_for_Domain_Adaptation_ICCV_2021_paper.pdf)  Semantic Concentration for Domain Adaptation, ICCV, 2021. </br>
[[Awais et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Awais_Adversarial_Robustness_for_Unsupervised_Domain_Adaptation_ICCV_2021_paper.pdf)  Adversarial Robustness for Unsupervised Domain Adaptation, ICCV, 2021.

#### Semi-supervised learning 
[[Sajjadi et al. 16]](https://arxiv.org/pdf/1606.04586.pdf) Regularization With Stochastic Transformations and Perturbations for Deep Semi-Supervised Learning, NIPS, 2016. </br>
[[Laine et al. 17]](https://arxiv.org/pdf/1610.02242.pdf)  Temporal Ensembling for Semi-Supervised Learning，ICLR, 2017. </br>
[[Tarvainen et al. 17]](https://arxiv.org/pdf/1703.01780.pdf) Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results, NIPS, 2017.  </br>
[[Miyato et al. 18]](https://arxiv.org/pdf/1704.03976.pdf) Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning, TPAMI, 2018.  </br>
[[Verma et al. 19]](https://arxiv.org/pdf/1903.03825.pdf) Interpolation Consistency Training for Semi-Supervised Learning, NIPS, 2019. </br>
[[Lee et al. 13]](url) Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks, ICML, 2013. </br>
[[Iscen et al. 19]](url) Label Propagation for Deep Semi-supervised Learning, CVPR, 2019.  </br>
[[Xie  et al. 20]](https://arxiv.org/pdf/1911.04252.pdf) Self-training with Noisy Student improves ImageNet classification, CVPR, 2020. </br>
[[Berthelot et al. 19]](https://arxiv.org/pdf/1905.02249.pdf) MixMatch: A Holistic Approach to Semi-Supervised Learning, NIPS, 2019. </br>
[[Berthelot et al. 20]](url) ReMixMatch: Semi-supervised learning with distribution alignment and augmentation anchoring, ICLR, 2020. </br>
[[Junnan Li et al. 20]](https://arxiv.org/pdf/2002.07394.pdf) DivideMix: Learning with Noisy Labels as Semi-supervised Learning, ICLR, 2020. </br>
[[Sohn et al. 20]](https://arxiv.org/ftp/arxiv/papers/2001/2001.07685.pdf) FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence, NIPS, 2020. </br>
[[Quali et al. 20]](https://arxiv.org/pdf/2006.05278.pdf)  An Overview of Deep Semi-Supervised Learning, 2020. </br>

--- 
[[Ke et al. 19]](https://arxiv.org/pdf/1909.01804.pdf) Dual Student: Breaking the Limits of the Teacher in Semi-supervised Learning, ICCV, 2019. </br>
[[Luo et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500766.pdf) Semi-supervised Semantic Segmentation via Strong-weak Dual-branch Network, ECCV, 2020. </br> 
[[Gao et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550511.pdf) Consistency-based Semi-supervised Active Learning: Towards Minimizing Labeling Cost, ECCV, 2020. </br>
[[Liu et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590307.pdf) Generative View-Correlation Adaptation for Semi-Supervised Multi-View Learning, ECCV, 2020. </br>
[[Kuo et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630460.pdf) FeatMatch: Feature-Based Augmentation for Semi-Supervised Learning, ECCV, 2020. </br>
[[Mustafa et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630579.pdf)  Transformation Consistency Regularization – A Semi-Supervised Paradigm for Image-to-Image Translation, ECCV, 2020. </br>
[[Chen et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Semi-Supervised_Semantic_Segmentation_With_Cross_Pseudo_Supervision_CVPR_2021_paper.pdf) Semi-Supervised Semantic Segmentation With Cross Pseudo Supervision,CVPR, 2021. </br>
[[Lai et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Lai_Semi-Supervised_Semantic_Segmentation_With_Directional_Context-Aware_Consistency_CVPR_2021_paper.pdf) Adaptive Consistency Regularization for Semi-Supervised Transfer Learning, CVPR,2021. </br>
[[Hu et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_SimPLE_Similar_Pseudo_Label_Exploitation_for_Semi-Supervised_Classification_CVPR_2021_paper.pdf) SimPLE: Similar Pseudo Label Exploitation for Semi-Supervised Classification,CVPR,2021. </br>
 [[Zhou et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_C3-SemiSeg_Contrastive_Semi-Supervised_Segmentation_via_Cross-Set_Learning_and_Dynamic_Class-Balancing_ICCV_2021_paper.pdf) Pixel Contrastive-Consistent Semi-Supervised Semantic Segmentation, ICCV, 2021. </br>
[[Xiong et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xiong_Multiview_Pseudo-Labeling_for_Semi-Supervised_Learning_From_Video_ICCV_2021_paper.pdf) Multiview Pseudo-Labeling for Semi-supervised Learning from Video, ICCV, 2021. </br>

### Image restoration and enhancement 

#### Image Deblurring 
[[Xu et al. 14]](https://papers.nips.cc/paper/2014/file/1c1d4df596d01da60385f0bb17a4a9e0-Paper.pdf) Deep Convolutional Neural Network for Image Deconvolution, NIPS, 2014. </br>
[[Zhang et al. 22]](https://arxiv.org/pdf/2201.10700.pdf) Deep Image Deblurring: A Survey, Arxiv, 2022. </br> 
[[Dong et al. 21]](https://arxiv.org/pdf/2103.09962.pdf) Deep Wiener Deconvolution: Wiener Meets Deep Learning for Image Deblurring, NIPS, 2021. </br> 
[[Nimisha et al., 17]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Nimisha_Blur-Invariant_Deep_Learning_ICCV_2017_paper.pdf) Blur-Invariant Deep Learning for Blind-Deblurring, ICCV, 2017. </br> 
[[Nah et al. 17]](https://arxiv.org/pdf/1612.02177.pdf) Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring, CVPR, 2017.  </br> 
[[Kupyn et al. 19]](https://arxiv.org/pdf/1908.03826.pdf) DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better, ICCV, 2019.  </br>
[[Zhang et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Deblurring_by_Realistic_Blurring_CVPR_2020_paper.pdf) Deblurring by Realistic Blurring, CVPR, 2020.  </br> 
[[Zhou et al. 19]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Spatio-Temporal_Filter_Adaptive_Network_for_Video_Deblurring_ICCV_2019_paper.pdf) Spatio-Temporal Filter Adaptive Network for Video Deblurring, ICCV, 2019.  </br> 
[[Nah et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Nah_Recurrent_Neural_Networks_With_Intra-Frame_Iterations_for_Video_Deblurring_CVPR_2019_paper.pdf) Recurrent Neural Networks with Intra-Frame Iterations for Video Deblurring, CVPR, 2019.  </br> 
[[Purohit et al. 20]](https://arxiv.org/pdf/1903.11394.pdf) Region-Adaptive Dense Network for Efficient Motion Deblurring, AAAI,2020. (SoTA of single image deblur on GoPro dataset) </br> 
[[Shen et al. 19]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shen_Human-Aware_Motion_Deblurring_ICCV_2019_paper.pdf)  Human-Aware Motion Deblurring, ICCV, 2019. </br> 

---- 

[[Rim et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700188.pdf)  Real-World Blur Dataset for Learning and Benchmarking Deblurring Algorithms, ECCV, 2020. </br> 
[[Lin et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530681.pdf) Learning Event-Driven Video Deblurring and Interpolation, ECCV, 2020.   </br> 
[[Zhong et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510188.pdf)  Efficient Spatio-Temporal Recurrent Neural Network for Video Deblurring, ECCV, 2020.  </br> 
[[Abuolaim et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550120.pdf)  Defocus Deblurring Using Dual-Pixel Data, ECCV, 2020.  </br> 
[[Cun et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580732.pdf) Defocus Blur Detection via Depth Distillation, ECCV, 2020.   </br> 
[[Chen et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Learning_a_Non-Blind_Deblurring_Network_for_Night_Blurry_Images_CVPR_2021_paper.pdf) Learning a Non-blind Deblurring Network for Night Blurry Images, CVPR, 2021.   </br> 
[[Rozumnyi et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Rozumnyi_DeFMO_Deblurring_and_Shape_Recovery_of_Fast_Moving_Objects_CVPR_2021_paper.pdf) DeFMO: Deblurring and Shape Recovery of Fast Moving Objects, CVPR, 2021.  </br> 
[[Xu et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Motion_Deblurring_With_Real_Events_ICCV_2021_paper.pdf)  Motion Deblurring with Real Events, ICCV, 2021.  </br> 
[[Cho et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Cho_Rethinking_Coarse-To-Fine_Approach_in_Single_Image_Deblurring_ICCV_2021_paper.pdf) Rethinking Coarse-to-Fine Approach in Single Image Deblurring, ICCV, 2021.  </br> 
[[Shang et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Shang_Bringing_Events_Into_Video_Deblurring_With_Non-Consecutively_Blurry_Frames_ICCV_2021_paper.pdf)  Bringing Events into Video Deblurring with Non-consecutively Blurry Frames, ICCV, 2021.   </br> 
[[Deng et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Deng_Multi-Scale_Separable_Network_for_Ultra-High-Definition_Video_Deblurring_ICCV_2021_paper.pdf) Multi-Scale Separable Network for Ultra-High-Definition Video Deblurring, ICCV, 2021.  </br> 
[[Hu et al 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Hu_Pyramid_Architecture_Search_for_Real-Time_Image_Deblurring_ICCV_2021_paper.pdf)  Pyramid Architecture Search for Real-Time Image Deblurring, ICCV, 2021.  </br> 

#### Image deraining 
[[Li et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Single_Image_Deraining_A_Comprehensive_Benchmark_Analysis_CVPR_2019_paper.pdf) Single Image Deraining: A Comprehensive Benchmark Analysis, CVPR, 2019. </br>
[[Li  et al. 21]](https://link.springer.com/content/pdf/10.1007/s11263-020-01416-w.pdf) A Comprehensive Benchmark Analysis of Single Image Deraining:
Current Challenges and Future Perspectives, IJCV, 2021.   </br> 
[[Yang et al. 17]](https://arxiv.org/pdf/1609.07769.pdf) Deep Joint Rain Detection and Removal from a Single Image, CVPR, 2017.  </br> 
[[Zhang et al. 18]](https://arxiv.org/pdf/1802.07412.pdf) Density-aware Single Image De-raining using a Multi-stream Dense Network, CVPR, 2018.  </br> 
[[Hu et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Depth-Attentional_Features_for_Single-Image_Rain_Removal_CVPR_2019_paper.pdf)   Depth-attentional features for single-image rain removal, CVPR, 2019.  </br> 
[[Qian et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Qian_Attentive_Generative_Adversarial_CVPR_2018_paper.pdf)  Attentive Generative Adversarial Network for Raindrop Removal from A Single Image, CVPR, 2018.  </br> 
[[Zhang et al. 19]](https://arxiv.org/pdf/1701.05957.pdf)  Image de-raining using a conditional generative adversarial network, IEEE transactions on circuits and systems for video technology, 2019.  </br> 
[[Wei et al. 19]](https://arxiv.org/pdf/1807.11078.pdf) Semi-supervised Transfer Learning for Image Rain Removal, CVPR, 2019.  </br> 
[[Yang et al. 17]](url) Deep Joint Rain Detection and Removal from a Single Image, CVPR, 2017. </br> 
[[Hu et al. 17]](url) Depth-Attentional Features for Single-Image Rain Removal, CVPR, 2019. </br> 

--- 
[[Yasarla et al. 20]](https://arxiv.org/pdf/2006.05580.pdf)  Syn2Real Transfer Learning for Image Deraining using Gaussian Processes, CVPR, 2020.   </br> 
[[Liu et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Unpaired_Learning_for_Deep_Image_Deraining_With_Rain_Direction_Regularizer_ICCV_2021_paper.pdf) Unpaired Learning for Deep Image Deraining with Rain Direction Regularizer, ICCV, 2021. </br> 
[[Zhou et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Image_De-Raining_via_Continual_Learning_CVPR_2021_paper.pdf)  Image De-raining via Continual Learning, CVPR, 2021.   </br> 
[[Wang et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Multi-Decoding_Deraining_Network_and_Quasi-Sparsity_Based_Training_CVPR_2021_paper.pdf) Multi-Decoding Deraining Network and Quasi-Sparsity Based Training, CVPR, 2021.  </br> 
[[Chen et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Robust_Representation_Learning_With_Feedback_for_Single_Image_Deraining_CVPR_2021_paper.pdf)  Robust Representation Learning with Feedback for Single Image Deraining, CVPR, 2021.  </br> 
[[Yue et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yue_Semi-Supervised_Video_Deraining_With_Dynamical_Rain_Generator_CVPR_2021_paper.pdf)  Semi-Supervised Video Deraining with Dynamical Rain Generator, CVPR, 2021.   </br> 
[[Yi et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yi_Structure-Preserving_Deraining_With_Residue_Channel_Prior_Guidance_ICCV_2021_paper.pdf) Structure-Preserving Deraining with Residue Channel Prior Guidance, ICCV,2021.   </br> 
[[Huang et a. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Memory_Oriented_Transfer_Learning_for_Semi-Supervised_Image_Deraining_CVPR_2021_paper.pdf) Memory Oriented Transfer Learning for Semi-Supervised Image Deraining, CVPR, 2021.  </br> 
[[Chen et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Pre-Trained_Image_Processing_Transformer_CVPR_2021_paper.pdf)  Pre-Trained Image Processing Transformer, CVPR, 2021.  </br> 
[[Jiang et al. 21]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_Multi-Scale_Progressive_Fusion_Network_for_Single_Image_Deraining_CVPR_2020_paper.pdf)  Multi-Scale Progressive Fusion Network for Single Image Deraining, CVPR, 2020.   </br> 
[[Fu et al. 20]](https://arxiv.org/pdf/1805.06173.pdf)  Lightweight Pyramid Networks for Image Deraining, IEEE Transactions on Neural Networks and Learning Systems, 2020. </br> 


### Image dehazing
[[Gui et al. 21]](https://arxiv.org/pdf/2106.03323.pdf) A Comprehensive Survey on Image Dehazing Based on Deep Learning, IJCAI, 2021.  </br> 
[[Cai et al. 16]](https://arxiv.org/pdf/1601.07661.pdf) DehazeNet: An End-to-End System for Single Image Haze Removal, IEEE, TIP, 2016. </br> 
[[Ren et al. 20]](https://link.springer.com/content/pdf/10.1007/s11263-019-01235-8.pdf) Single Image Dehazing via Multi-scale Convolutional Neural Networks
with Holistic Edges, IJCV, 2020. (Extension of the conference version at 2016) </br> 
[[Li et al. 17]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_AOD-Net_All-In-One_Dehazing_ICCV_2017_paper.pdf) AOD-Net: All-in-One Dehazing Network, ICCV, 2017. </br> 
[[Qin et al. 20]](https://arxiv.org/pdf/1911.07559.pdf) FFA-Net: Feature Fusion Attention Network for Single Image Dehazing, AAAI,2020. </br> 
[[Zhang et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Densely_Connected_Pyramid_CVPR_2018_paper.pdf) Densely Connected Pyramid Dehazing Network, CVPR, 2018. </br> 
[[Ren et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ren_Gated_Fusion_Network_CVPR_2018_paper.pdf)  Gated Fusion Network for Single Image Dehazing
, CVPR, 2018. </br> 
[[Qu et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Qu_Enhanced_Pix2pix_Dehazing_Network_CVPR_2019_paper.pdf) Enhanced Pix2pix Dehazing Network, CVPR, 2019.  </br> 
[[Hong et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hong_Distilling_Image_Dehazing_With_Heterogeneous_Task_Imitation_CVPR_2020_paper.pdf)  Distilling Image Dehazing With Heterogeneous Task Imitation, CVPR, 2020.  </br> 
[[Shao et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shao_Domain_Adaptation_for_Image_Dehazing_CVPR_2020_paper.pdf) Domain Adaptation for Image Dehazing, CVPR, 2020.  </br> 
[[Engin et al. 18]]( https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Engin_Cycle-Dehaze_Enhanced_CycleGAN_CVPR_2018_paper.pdf )Cycle-Dehaze: Enhanced CycleGAN for Single Image Dehazing, ECCVW, 2018. </br> 
[[Li et al. 20]](https://ieeexplore-ieee-org.lib.ezproxy.ust.hk/stamp/stamp.jsp?tp=&arnumber=9170880&tag=1) Zero-Shot Image Dehazing, IEEE TIP, 2020.  </br> 

--- 
[[Wu et al. 21]](https://arxiv.org/pdf/2104.09367.pdf) Contrastive Learning for Compact Single Image Dehazing, CVPR, 2021. </br> 
[[Shyam et al. 21]](https://arxiv.org/pdf/2101.10449.pdf) Towards Domain Invariant Single Image Dehazing, AAAI, 2021. </br> 
[[Zheng et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Ultra-High-Definition_Image_Dehazing_via_Multi-Guided_Bilateral_Learning_CVPR_2021_paper.pdf) Ultra-High-Defifinition Image Dehazing via Multi-Guided Bilateral Learning, CVPR, 2021. </br> 
[[Chen et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_PSD_Principled_Synthetic-to-Real_Dehazing_Guided_by_Physical_Priors_CVPR_2021_paper.pdf) PSD: Principled Synthetic-to-Real Dehazing Guided by Physical Priors, CVPR, 2021. </br> 
[[Zhao et al. 21]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_BidNet_Binocular_Image_Dehazing_Without_Explicit_Disparity_Estimation_CVPR_2020_paper.pdf) BidNet: Binocular Image Dehazing without Explicit Disparity Estimation, CVPR, 2021.   </br> 
[[Kar et al. 21]](http://xxx.itp.ac.cn/pdf/2008.01701v1) Transmission Map and Atmospheric Light Guided Iterative Updater Network for Single Image Dehazing, CVPR, 2021. </br> 
[[Li et al. 20]](https://ieeexplore.ieee.org/abstract/document/8902220) Semi-Supervised Image Dehazing, IEEE TIP, 2020.  </br> 
[[Yi et al. 21]](https://arxiv.org/pdf/2102.03501.pdf)  Two-Step Image Dehazing with Intra-domain and Inter-domain Adaptation, Arxiv, 2021.

####  Image/Video Super-Resolution 
[[Dong et al. 16]](https://arxiv.org/pdf/1501.00092.pdf) mage Super-Resolution Using Deep Convolutional Networks, ECCV,2016.(First deep learning-based method)  </br>
[[Lim et al. 17]](https://arxiv.org/pdf/1707.02921.pdf) Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW, 2017.  </br> 
[[Wang et al. 19]](https://arxiv.org/pdf/1902.06068.pdf)  Deep Learning for Image Super-resolution: A Survey, IEEE TPAMI, 2021. </br> 
[[Kim et al. 17]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Kim_Accurate_Image_Super-Resolution_CVPR_2016_paper.pdf) Accurate Image Super-Resolution Using Very Deep Convolutional Networks, CVPR, 2017.  </br> 
[[Tai et al. 17]](https://arxiv.org/pdf/1708.02209.pdf) MemNet: A Persistent Memory Network for Image Restoration, CVPR, 2017.  </br> 
[[Li et al. 18]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf)  Multi-scale Residual Network for Image Super-Resolution, ECCV, 2018. </br> 
[[Zhang et al. 18]](https://arxiv.org/pdf/1807.02758.pdf) Image Super-Resolution Using Very Deep Residual Channel Attention Networks, ECCV, 2018. </br> 
[[Zhang et al. 19]](https://arxiv.org/pdf/1903.10082.pdf) Residual Non-local Attention Networks for Image Restoration, ICLR, 2019. </br> 
[[Dai et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.pdf) Second-order Attention Network for Single Image Super-Resolution, CVPR, 2019. </br> 
[[Han et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/3225.pdf)  Image Super-Resolution via Dual-State Recurrent Networks, CVPR, 2018. </br> 
[[Li et al. 18]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf) Multi-scale Residual Network for Image Super-Resolution, ECCV, 2018. </br> 
[[Ren et al. 18]](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Ren_Image_Super_Resolution_CVPR_2017_paper.pdf) Image Super Resolution Based on Fusing Multiple Convolution Neural Networks, CVPRW, 2017. </br> 
[[Ahn et al. 18]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Namhyuk_Ahn_Fast_Accurate_and_ECCV_2018_paper.pdf) Fast, accurate, and lightweight
super-resolution with cascading residual network, ECCV, 2018. </br> 
[[Zhang et al. 19]](https://ieeexplore-ieee-org.lib.ezproxy.ust.hk/stamp/stamp.jsp?tp=&arnumber=8502129&tag=1)  DCSR: Dilated Convolutions for Single Image Super-Resolution, IEEE TIP, 2019. </br> 
[[Zhantg et al. 18]](https://arxiv.org/pdf/1802.08797.pdf) Residual Dense Network for Image Super-Resolution, CVPR, 2018. </br> 
[[Hu et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Meta-SR_A_Magnification-Arbitrary_Network_for_Super-Resolution_CVPR_2019_paper.pdf) Meta-SR: A Magnification-Arbitrary Network for Super-Resolution, CVPR, 2021. </br> 
[[Chen et al. 21]](https://arxiv.org/pdf/2012.09161.pdf)  Learning Continuous Image Representation with Local Implicit Image Function, CVPR, 2021. </br> 
[[Lee et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690460.pdf) Learning with Privileged Information for Efficient Image Super-Resolution, ECCV, 2020. </br> 
[[Hu et al. 21]](https://www.ijcai.org/proceedings/2021/0155.pdf) Towards Compact Single Image Super-Resolution via Contrastive Self-distillation, IJCAI, 2021.  </br> 
[[Cai et al. 19]](https://csjcai.github.io/papers/RealSR.pdf) Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Model, ICCV, 2019. </br> 
[[Wei et al. 20]](https://arxiv.org/pdf/2008.01928.pdf) Component Divide-and-Conquer for Real-World Image Super-Resolution, ECCV, 2021. </br> 
[[Wang et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Unsupervised_Real-World_Super-Resolution_A_Domain_Adaptation_Perspective_ICCV_2021_paper.pdf) Unsupervised Real-World Super-Resolution: A Domain Adaptation Perspective, ICCV, 2021. </br> 
[[Maeda et a. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Maeda_Unpaired_Image_Super-Resolution_Using_Pseudo-Supervision_CVPR_2020_paper.pdf) Unpaired Image Super-Resolution using Pseudo-Supervision, CVPR, 2020. </br> 
[[Shocher et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Shocher_Zero-Shot_Super-Resolution_Using_CVPR_2018_paper.pdf) “Zero-Shot” Super-Resolution using Deep Internal Learning, CVPR, 2018. </br> 

----
[[Wei et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wei_Unsupervised_Real-World_Image_Super_Resolution_via_Domain-Distance_Aware_Training_CVPR_2021_paper.pdf) Unsupervised Real-World Super-Resolution: A Domain Adaptation Perspective, ICCV, 2021. </br> 
[[Zhang et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Data-Free_Knowledge_Distillation_for_Image_Super-Resolution_CVPR_2021_paper.pdf) Unsupervised Real-world Image Super Resolution via Domain-distance Aware Training, CVPR, 2021.   </br> 
[[Sefi et al. 20]](https://arxiv.org/pdf/1909.06581.pdf) Blind Super-Resolution Kernel Estimation using an Internal-GAN, NIPS, 2020.  </br> 
[[Cheng et a. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620256.pdf)  Zero-Shot Image Super-Resolution with Depth Guided Internal Degradation Learning, ECCV, 2020.  </br> 
[[Sun et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_Learning_Scene_Structure_Guidance_via_Cross-Task_Knowledge_Transfer_for_Single_CVPR_2021_paper.pdf) Learning Scene Structure Guidance via Cross-Task Knowledge Transfer for Single Depth Super-Resolution, CVPR, 2021.  </br> 
[[Wang et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Unsupervised_Degradation_Representation_Learning_for_Blind_Super-Resolution_CVPR_2021_paper.pdf) Unsupervised Degradation Representation Learning for Blind Super-Resolution, CVPR, 2021.  </br> 
[[Son et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Son_SRWarp_Generalized_Image_Super-Resolution_under_Arbitrary_Transformation_CVPR_2021_paper.pdf) SRWarp: Generalized Image Super-Resolution under Arbitrary Transformation, CVPR, 2021.  </br> 
[[Jo et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Jo_Tackling_the_Ill-Posedness_of_Super-Resolution_Through_Adaptive_Target_Generation_CVPR_2021_paper.pdf) Tackling the Ill-Posedness of Super-Resolution through Adaptive Target Generation, CVPR, 2021.  </br> 
[[Mei et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Mei_Image_Super-Resolution_With_Non-Local_Sparse_Attention_CVPR_2021_paper.pdf) Image Super-Resolution with Non-Local Sparse Attention, CVPR, 2021.  </br> 
[[Wang et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Learning_a_Single_Network_for_Scale-Arbitrary_Super-Resolution_ICCV_2021_paper.pdf) Learning a Single Network for Scale-Arbitrary Super-Resolution, ICCV, 2021.  </br> 
[[Wang et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Dual-Camera_Super-Resolution_With_Aligned_Attention_Modules_ICCV_2021_paper.pdf) Dual-Camera Super-Resolution with Aligned Attention Modules, CVPR, 2021. </br> 
[[Chan et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chan_BasicVSR_The_Search_for_Essential_Components_in_Video_Super-Resolution_and_CVPR_2021_paper.pdf) BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond, ICCV, 2021. </br> 
[[Yi et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yi_Omniscient_Video_Super-Resolution_ICCV_2021_paper.pdf)  Omniscient Video Super-Resolution, ICCV, 2021. </br> 
[[Tian et al. 20]](https://arxiv.org/pdf/1812.02898.pdf) TDAN: Temporally Deformable Alignment Network for Video Super-Resolution, CVPR, 2020. </br> 
[[Wang et al. 19]](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Wang_EDVR_Video_Restoration_With_Enhanced_Deformable_Convolutional_Networks_CVPRW_2019_paper.pdf)  EDVR: Video Restoration With Enhanced Deformable Convolutional Networks, CVPRW, 2019. </br> 
 
#### Deep HDR imaging 

[[Wang et al. 21]](https://arxiv.org/pdf/2110.10394.pdf)  Deep Learning for HDR Imaging:State-of-the-Art and Future Trends, IEEE TPAMI, 2021. </br> 
[[Kalantrai et al. 17]](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR.pdf) Deep High Dynamic Range Imaging of Dynamic Scenes, Siggraph, 2017. </br> 
[[Prabhakar et al. 19]](https://ieeexplore.ieee.org/document/8747329) A Fast, Scalable, and Reliable Deghosting Method for Extreme Exposure Fusion, ICCP, 2019. </br> 
[[Wu et al. 18]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Shangzhe_Wu_Deep_High_Dynamic_ECCV_2018_paper.pdf) Deep High Dynamic Range Imaging with Large Foreground Motions, ECCV, 2018. </br> 
[[Yan et al. 21]](https://www.sciencedirect.com/science/article/abs/pii/S092523122031849X) Towards accurate HDR imaging with learning generator constraints, Neurocomputing, 2020. </br> 
[[Yan et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yan_Attention-Guided_Network_for_Ghost-Free_High_Dynamic_Range_Imaging_CVPR_2019_paper.pdf)  Attention-guided Network for Ghost-free High Dynamic Range Imaging, CVPR, 2019. </br> 
[[Rosh et al. 19]](https://ieeexplore-ieee-org.lib.ezproxy.ust.hk/stamp/stamp.jsp?tp=&arnumber=8803582) Deep Multi-Stage Learning for HDR With Large Object Motions, ICCP, 2019. </br> 
[[Xu et al. 20]](https://ieeexplore-ieee-org.lib.ezproxy.ust.hk/stamp/stamp.jsp?tp=&arnumber=9112609) MEF-GAN: Multi-Exposure Image Fusion via Generative Adversarial Networks, TIP, 2020. </br> 
[[Eilertsen et al. 17]](https://arxiv.org/pdf/1710.07480.pdf)  HDR image reconstruction from a single exposure using deep CNNs, Siggraph, 2017. </br> 
[[Santas et al. 20]](https://arxiv.org/pdf/2005.07335.pdf) Single Image HDR Reconstruction Using a CNN with Masked Features and Perceptual Loss, Siggraph, 2020. </br> 
[[Endo et al. 17]](http://www.cgg.cs.tsukuba.ac.jp/~endo/projects/DrTMO/paper/DrTMO_SIGGRAPHAsia_light.pdf) Deep Reverse Tone Mapping, Siggraph, 2017. </br> 
[[Liu et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Single-Image_HDR_Reconstruction_by_Learning_to_Reverse_the_Camera_Pipeline_CVPR_2020_paper.pdf) Single-Image HDR Reconstruction by Learning to Reverse the Camera Pipeline, CVPR, 2020. </br> 

---
[[Metzler]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Metzler_Deep_Optics_for_Single-Shot_High-Dynamic-Range_Imaging_CVPR_2020_paper.pdf) Deep Optics for Single-shot High-dynamic-range Imaging, CVPR, 2020. </br> 
[[Kim et al. 18]](https://link.springer.com/content/pdf/10.1007%2F978-3-030-20893-6_24.pdf) A Multi-purpose Convolutional Neural Network for Simultaneous Super-Resolution and High Dynamic Range Image Reconstruction, ACCV, 2018. </br> 
[[Kim et al. 19]](https://arxiv.org/ftp/arxiv/papers/1904/1904.11176.pdf) Deep sr-itm: Joint learning of superresolution and inverse tone-mapping for 4k uhd hdr applications, ICCV,2019. </br> 
[[Kim et al. 20]](https://arxiv.org/pdf/1909.04391.pdf)  JSI-GAN: GAN-Based Joint Super-Resolution and Inverse Tone-Mapping with Pixel-Wise Task-Specific Filters for UHD HDR Video, AAAI, 2020. </br> 
[[Kim et al. 20]](https://arxiv.org/pdf/2006.15833.pdf) End-to-End Differentiable Learning to HDR Image Synthesis for Multi-exposure Images, AAAI, 2020. </br> 
[[Chen et al. 21]](https://arxiv.org/pdf/2103.14943.pdf)  HDR Video Reconstruction: A Coarse-to-fine Network and A Real-world Benchmark Dataset, ICCV, 2021. </br> 
[[Jiang et al. 21]](https://arxiv.org/pdf/2103.10982.pdf) HDR Video Reconstruction with Tri-Exposure Quad-Bayer Sensors, Arxiv, 2021. </br> 


### Object detection
[[Fang et al. 21]](https://arxiv.org/pdf/2106.00666.pdf) You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection, NIPS, 2021. </br> 
[[Ge et al. 21]](https://arxiv.org/pdf/2107.08430.pdf)  YOLOX: Exceeding YOLO Series in 2021, Arxiv, 2021. </br> 
[[Pramanik et al. 22]](https://ieeexplore.ieee.org/document/9313052) Granulated RCNN and Multi-Class Deep SORT for Multi-Object Detection and Tracking, IEEE TETCI, 2022. </br> 
[[Wang et al. 21]](https://arxiv.org/pdf/2105.04206.pdf)  You Only Learn One Representation: Unified Network for Multiple Tasks, Arxiv, 2021. </br> 
[[]](url)

---
#### Face detection 
[[Luo et al. 16]](https://proceedings.neurips.cc/paper/2016/file/c8067ad1937f728f51288b3eb986afaa-Paper.pdf) Understanding the Effective Receptive Field in Deep Convolutional Neural Networks, 2016. </br> 
[[Tang et al. 18]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xu_Tang_PyramidBox_A_Context-assisted_ECCV_2018_paper.pdf) PyramidBox: A Context-assisted Single Shot Face Detector, ECCV, 2018. </br> 
[[Liu et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_High-Level_Semantic_Feature_Detection_A_New_Perspective_for_Pedestrian_Detection_CVPR_2019_paper.pdf) High-level Semantic Feature Detection: A New Perspective for Pedestrian Detection, CVPR, 2019. 
[[Li et al. 20]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_DSFD_Dual_Shot_Face_Detector_CVPR_2019_paper.pdf) Dsfd: Dual shot face detector， CVPR, 2019. </br> 
[[Wang et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Hierarchical_Pyramid_Diverse_Attention_Networks_for_Face_Recognition_CVPR_2020_paper.pdf) Hierarchical Pyramid Diverse Attention Networks for Face Recognition, CVPR, 2020. </br> 
[[Huang et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_When_Age-Invariant_Face_Recognition_Meets_Face_Age_Synthesis_A_Multi-Task_CVPR_2021_paper.pdf) When Age-Invariant Face Recognition Meets Face Age Synthesis: A Multi-Task Learning Framework, CVPR, 2021. </br> 
[[Tong et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Tong_FaceSec_A_Fine-Grained_Robustness_Evaluation_Framework_for_Face_Recognition_Systems_CVPR_2021_paper.pdf) FACESEC: A Fine-grained Robustness Evaluation Framework for Face Recognition Systems, CVPR, 2021. </br> 
[[Qiu et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Qiu_SynFace_Face_Recognition_With_Synthetic_Data_ICCV_2021_paper.pdf) SynFace: Face Recognition with Synthetic Data, ICCV, 2021. </br> 
[[Song et al. 21]](https://openaccess.thecvf.com/content/ICCV2021W/MFR/papers/Huang_Masked_Face_Recognition_Datasets_and_Validation_ICCVW_2021_paper.pdf)  Occlusion Robust Face Recognition Based on Mask Learning With Pairwise Differential Siamese Network, ICCV, 2021. </br> 
[[Fabbri et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Fabbri_MOTSynth_How_Can_Synthetic_Data_Help_Pedestrian_Detection_and_Tracking_ICCV_2021_paper.pdf) MOTSynth: How Can Synthetic Data Help Pedestrian Detection and Tracking?, ICCV, 2021.

#### Pedestrain detection
[[Wang et al. 18]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.pdf)  Repulsion Loss: Detecting Pedestrians in a Crowd, CVPR, 2018.  </br> 
[[Zhang et al. 18]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Shifeng_Zhang_Occlusion-aware_R-CNN_Detecting_ECCV_2018_paper.pdf)  Occlusion-aware R-CNN: Detecting Pedestrians in a Crowd, ECCV, 2018.  </br> 
[[Liu et al. 19]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Adaptive_NMS_Refining_Pedestrian_Detection_in_a_Crowd_CVPR_2019_paper.pdf) Adaptive NMS: Refining Pedestrian Detection in a Crowd, CVPR, 2019.  </br> 
[[Zhou et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630766.pdf) Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems, ECCV, 2020. </br> 
[[Wu et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Temporal-Context_Enhanced_Detection_of_Heavily_Occluded_Pedestrians_CVPR_2020_paper.pdf) Temporal-Context Enhanced Detection of Heavily Occluded Pedestrians, CVPR, 2020.   </br> 
[[Wu  et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Luo_Where_What_Whether_Multi-Modal_Learning_Meets_Pedestrian_Detection_CVPR_2020_paper.pdf) Where, What, Whether: Multi-modal Learning Meets Pedestrian Detection, CVPR, 2020.  </br> 
[[Huang et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_NMS_by_Representative_Region_Towards_Crowded_Pedestrian_Detection_by_Proposal_CVPR_2020_paper.pdf) NMS by Representative Region: Towards Crowded Pedestrian Detection by Proposal Pairing, CVPR, 2020.  </br> 
[[Wang et al. 20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Learning_Human-Object_Interaction_Detection_Using_Interaction_Points_CVPR_2020_paper.pdf) Learning Human-Object Interaction Detection using Interaction Points, CVPR, 2020.  </br> 
[[Sundararaman et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Sundararaman_Tracking_Pedestrian_Heads_in_Dense_Crowd_CVPR_2021_paper.pdf) Tracking Pedestrian Heads in Dense Crowd, CVPR, 2020.  </br> 
[[Yan et al. 20]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_Anchor-Free_Person_Search_CVPR_2021_paper.pdf)  Anchor-Free Person Search, CVPR,2020.  </br> 



### Computer vision with novel camera sensors (II)
[[Ghose et al. 19]](https://openaccess.thecvf.com/content_CVPRW_2019/papers/PBVS/Ghose_Pedestrian_Detection_in_Thermal_Images_Using_Saliency_Maps_CVPRW_2019_paper.pdf) Pedestrian Detection in Thermal Images using Saliency Maps, CVPR, 2019.   </br> 
[[Kieu et al. 20]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670545.pdf) Task-conditioned Domain Adaptation for Pedestrian Detection in Thermal Imagery, ECCV, 2020. </br> 

### Scene Understanding in adverse vision conditions
[[Wang et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_HLA-Face_Joint_High-Low_Adaptation_for_Low_Light_Face_Detection_CVPR_2021_paper.pdf)  HLA-Face: Joint High-Low Adaptation for Low Light Face Detection, CVPR, 2021.
