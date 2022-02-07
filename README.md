# Deep Learning for Visual Intelligence: Trends and Challenges 

## Course information

* Instructor: WANG Lin (linwang@ust.hk)
* TAs: SU Ying (ysuay@connect.ust.hk) and ZHU Qingyan (qzhuai@connect.ust.hk)
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
[[Yang et al. 21]](https://arxiv.org/pdf/2111.10346.pdf) Global and Local Alignment Networks for Unpaired Image-to-Image Translation, Arxiv 2021. 

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
####  Image/Video Super-Resolution 
[[Chan et al. 21]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chan_BasicVSR_The_Search_for_Essential_Components_in_Video_Super-Resolution_and_CVPR_2021_paper.pdf) BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond, ICCV, 2021. </br> 
[[Yi et al. 21]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yi_Omniscient_Video_Super-Resolution_ICCV_2021_paper.pdf)  Omniscient Video Super-Resolution, ICCV, 2021. </br> 


