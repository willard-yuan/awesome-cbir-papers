<div align="center">
	<img width="500" height="350" src="logo.svg" alt="Awesome">
	<br>
  <p>
    <a href="https://github.com/willard-yuan/awesome-cbir-papers">CBIR in academia and industry</a>
  </p>
</div>

# Awesome image retrieval papers

The main goal is to collect classical and solid works of image retrieval in academia and industry.

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

- [Classical Local Feature](#classical-local-feature)
- [Deep Learning Feature (Global Feature)](#deep-learning-feature-global-feature)
- [Deep Learning Feature (Local Feature)](#deep-learning-feature-local-feature)
- [Deep Learning Feature (Instance Search)](#deep-learning-feature-instance-search)
- [ANN search](#ann-search)
- [CBIR Attack](#cbir-attack)
- [CBIR rank](#cbir-rank)
- [CBIR in Industry](#cbir-in-industry)
- [CBIR Competition and Challenge](#cbir-competition-and-challenge)
- [CBIR for Duplicate(copy) detection](#cbir-for-duplicatecopy-detection)
- [Feature Fusion](#feature-fusion)
- [Instance Matching](#instance-matching)
- [Semantic Matching](#semantic-matching)
- [Template Matching](#template-matching)
- [Image Identification](#image-identification)
- [Tutorials](#tutorials)
- [Slide](#slide)
- [Demo and Demo Online](#demo-and-demo-online)
- [Datasets](#datasets)
- [Useful Package](#useful-package)

## Classical Local Feature

- [Object retrieval with large vocabularies and fast spatial matching](https://www.robots.ox.ac.uk/~vgg/publications/papers/philbin07.pdf), CVPR 2007.
- [Visual Categorization with Bags of Keypoints](http://www.cs.princeton.edu/courses/archive/fall09/cos429/papers/csurka-eccv-04.pdf), ECCV 2004.
- [ORB: an efficient alternative to SIFT or SURF](https://www.willowgarage.com/sites/default/files/orb_final.pdf), ICCV 2011.
- [Object Recognition from Local Scale-Invariant Features](http://www.cs.ubc.ca/~lowe/papers/iccv99.pdf), ICCV 1999.
- [Total Recall: Automatic Query Expansion with a Generative Feature Model for Object Retrieval](https://www.robots.ox.ac.uk/~vgg/publications/papers/philbin07.pdf), ICCV 2007.
- [Three things everyone should know to improve object retrieval](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf), CVPR 2012.
- [On-the-fly learning for visual search of large-scale image and video datasets](https://www.robots.ox.ac.uk/~vgg/publications/2015/Chatfield15/chatfield15.pdf)
- [All about VLAD](), CVPR 2013.
- [Aggregating localdescriptors into a compact image representation](https://lear.inrialpes.fr/pubs/2010/JDSP10/jegou_compactimagerepresentation.pdf), CVPR 2010.
- [More About VLAD: A Leap from Euclidean to Riemannian Manifolds](https://paperswithcode.com/paper/more-about-vlad-a-leap-from-euclidean-to), CVPR 2015.
- [Hamming embedding and weak geometric consistency for large scale image search](https://lear.inrialpes.fr/pubs/2008/JDS08/jegou_hewgc08.pdf), CVPR 2008.
- [Revisiting the VLAD image representation](https://hal.inria.fr/hal-00840653v1/document), [project](https://github.com/jorjasso/VLAD/blob/master/VLADlib/VLAD.py)
- [Improving the Fisher Kernel for Large-Scale Image Classification](https://www.robots.ox.ac.uk/~vgg/rg/papers/peronnin_etal_ECCV10.pdf), ECCV 2010.
- [Image Classification with the Fisher Vector: Theory and Practice](https://hal.inria.fr/hal-00830491/document)
- [Democratic Diffusion Aggregation for ImageRetrieval]()
- [A Vote-and-Verify Strategy for Fast Spatial Verification in Image Retrieval](https://www.microsoft.com/en-us/research/uploads/prod/2019/09/accv_2016_schoenberger.pdf), ACCV 2016.
- [Triangulation embedding and democratic aggregation for image search](https://www.robots.ox.ac.uk/~vgg/publications/2014/Jegou14/jegou14.pdf), CVPR 2014.
- [Efficient Large-scale Image Search With a Vocabulary Tree](http://www.ipol.im/pub/art/2018/199/), IPOL 2015, [code](https://github.com/fragofer/voctree).

## Deep Learning Feature (Global Feature)

- [Online Invariance Selection for Local Feature Descriptors](https://arxiv.org/abs/2007.08988), ECCV 2020, [code](https://github.com/rpautrat/LISRD).
- [Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval](https://arxiv.org/pdf/2007.12163.pdf), ECCV 2020.
- [SOLAR: Second-Order Loss and Attention for Image Retrieval](https://arxiv.org/pdf/2001.08972.pdf), ECCV 2020.
- [Unifying Deep Local and Global Features for Image Search](https://arxiv.org/abs/2001.05027), arxiv 2020.
- [SOLAR: Second-Order Loss and Attention for Image Retrieval](https://arxiv.org/abs/2001.08972v2), arxiv 2020.
- [A Benchmark on Tricks for Large-scale Image Retrieval](https://arxiv.org/pdf/1907.11854.pdf)，arxiv 2020.
- [Learning with Average Precision: Training Image Retrieval with a Listwise Loss](https://arxiv.org/pdf/1906.07589v1.pdf), ICCV 2019.
- [MultiGrain: a unified image embedding for classes and instances](https://arxiv.org/abs/1902.05509), arxiv 2019.
- [Deep Image Retrieval:Learning Global Representations for Image search](https://arxiv.org/abs/1604.01325).
- [End-to-end Learning of Deep Visual Representations for Image retrieval](https://arxiv.org/abs/1610.07940), DIR更详细的论文说明.
- [What Is the Best Practice for CNNs Applied to Visual Instance Retrieval?](https://arxiv.org/abs/1611.01640), 关于layer选取的问题.
- [Bags of Local Convolutional Features for Scalable Instance Search](https://arxiv.org/abs/1604.01325).
- [Faster R-CNN Features for Instance Search](https://github.com/imatge-upc/retrieval-2016-deepvision), CVPR workshop 2016.
- [Cross-dimensional Weighting for Aggregated Deep Convolutional Features](https://arxiv.org/abs/1512.04065), [project](https://github.com/yahoo/crow).
- [Class-Weighted Convolutional Features for Image Retrieval](https://github.com/imatge-upc/retrieval-2017-cam).
- [Multi-Scale Orderless Pooling of Deep Convolutional Activation Features](), VLAD coding.
- [Aggregating Deep Convolutional Features for Image Retrieval](https://arxiv.org/abs/1510.07493), [论文笔记](https://zhuanlan.zhihu.com/p/23136747), [基于深度学习的视觉实例搜索研究进展](https://zhuanlan.zhihu.com/p/22265265).
- [Particular object retrieval with integral max-pooling of CNN activations](https://arxiv.org/abs/1511.05879), [project](http://cmp.felk.cvut.cz/~toliageo/soft.html).
- [Particular object retrieval using CNN](https://github.com/AaltoVision/Object-Retrieval).
- [Learning to Match Aerial Images with Deep Attentive Architectures](https://vision.cornell.edu/se3/wp-content/uploads/2016/04/1204.pdf).
- [Siamese Network of Deep Fisher-Vector Descriptors for Image Retrieval](https://arxiv.org/pdf/1702.00338v1.pdf).
- [Combining Fisher Vector and Convolutional Neural Networks for Image Retrieval](http://ceur-ws.org/Vol-1653/paper_19.pdf), fv和cnn特征融合提升.
- [Selective Deep Convolutional Features for Image Retrieval](https://arxiv.org/pdf/1707.00809v1.pdf), ACM MM 2017.
- [Class-Weighted Convolutional Features for Image Retrieval](https://github.com/imatge-upc/retrieval-2017-cam).
- [Fine-tuning CNN Image Retrieval with No Human Annotation](https://arxiv.org/abs/1711.02512), TPAMI 2018.
- [An accurate retrieval through R-MAC+ descriptors for landmark recognition](https://arxiv.org/pdf/1806.08565.pdf).
- [Regional Attention Based Deep Feature for Image Retrieval](https://sglab.kaist.ac.kr/RegionalAttention/), [code](https://github.com/jaeyoon1603/Retrieval-RegionalAttention), BMVC 2018.
- [Detect-to-Retrieve: Efficient Regional Aggregation for Image Search](https://arxiv.org/pdf/1812.01584.pdf), CVPR 2019.
- [Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking](http://cmp.felk.cvut.cz/~toliageo/p/RadenovicIscenToliasAvrithisChum_CVPR2018_Revisiting%20Oxford%20and%20Paris:%20Large-Scale%20Image%20Retrieval%20Benchmarking.pdf), [project](http://cmp.felk.cvut.cz/revisitop/), CVPR 2018.
- [Guided Similarity Separation for Image Retrieval](https://github.com/layer6ai-labs/GSS), NeurIPS 2019.

## Deep Learning Feature (Local Feature)

- [COTR: Correspondence Transformer for Matching Across Images](https://github.com/ubc-vision/COTR), arxiv 2021.
- [Online Invariance Selection for Local Feature Descriptors](https://arxiv.org/abs/2007.08988), ECCV 2020, [code](https://github.com/rpautrat/LISRD).
- [Learning and aggregating deep local descriptors for instance-level recognition](https://arxiv.org/abs/2007.13172), ECCV 2020, [code](https://github.com/gtolias/how).
- [DISK: Learning local features with policy gradient](https://arxiv.org/pdf/2006.13566.pdf), NeurIPS 2020, [code](https://github.com/cvlab-epfl/disk).
- [Learning and aggregating deep local descriptorsfor instance-level recognition](https://paperswithcode.com/paper/learning-and-aggregating-deep-local/review/), ECCV 2020, [code](https://github.com/jenicek/asmk).
- [D2D: Keypoint Extraction with Describe to Detect Approach](https://arxiv.org/pdf/2005.13605.pdf), arxiv 2020.
- [UR2KiD: Unifying Retrieval, Keypoint Detection, and Keypoint Description without Local Correspondence Supervision](https://arxiv.org/abs/2001.07252), arxiv.
- [Visualizing Deep Similarity Networks](https://arxiv.org/pdf/1901.00536.pdf), WACV 2019.
- [Combination of Multiple Global Descriptors for Image Retrieval](https://github.com/naver/cgd).
- [Beyond Cartesian Representations for Local Descriptors](https://arxiv.org/abs/1908.05547), [code](https://github.com/cvlab-epfl/log-polar-descriptors), ICCV 2019.
- [R2D2: Reliable and Repeatable Detector and Descriptor](https://arxiv.org/abs/1906.06195), [code](https://github.com/naver/r2d2), NeurIPS 2019.
- [SOSNet: Second Order Similarity Regularization for Local Descriptor Learning](https://github.com/scape-research/SOSNet), CVPR 2019.
- [Local Features and Visual Words Emerge in Activations](https://avrithis.net/data/pub/pdf/conf/C110.cvpr19.spatial.pdf), CVPR 2019.
- [Explicit Spatial Encoding for Deep Local Descriptors](https://arxiv.org/abs/1904.07190), CVPR 2019.
- [Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters](https://github.com/axelBarroso/Key.Net), ICCV 2019.
- [Learning Discriminative Affine Regions via Discriminability](http://cn.arxiv.org/pdf/1711.06704.pdf), [affnet](https://github.com/ducha-aiki/affnet).
- [A Large Dataset for Improving Patch Matching](http://cn.arxiv.org/pdf/1801.01466.pdf), [PS-Dataset](https://github.com/rmitra/PS-Dataset).
- [Working hard to know your neighbor's margins: Local descriptor learning loss](), [code](https://github.com/DagnyT/hardnet).
- [MatchNet: Unifying Feature and Metric Learning for Patch-Based Matching](), [code](https://github.com/hanxf/matchnet).
- [LF-Net: Learning Local Features from Images](https://arxiv.org/abs/1805.09662), NeurIPS 2018.
- [Local Descriptors Optimized for Average Precision](http://openaccess.thecvf.com/content_cvpr_2018/papers/He_Local_Descriptors_Optimized_CVPR_2018_paper.pdf), CVPR 2018.
- [SuperPoint: Self-Supervised Interest Point Detection and Description](http://cn.arxiv.org/pdf/1712.07629.pdf), Magic Leap.
- [GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints](https://arxiv.org/pdf/1807.06294.pdf), [code](https://github.com/lzx551402/geodesc), ECCV 2018.
- [Learning local feature descriptors with triplets and shallow convolutional neural networks](https://github.com/vbalnt/tfeat), BMVC 2016.
  

## Deep Learning Feature (Instance Search)

- [Deeply Activated Salient Region for Instance Search](https://arxiv.org/abs/2002.00185), arXiv 2020.
- [Instance search based on weakly supervised feature learning](https://doi.org/10.1016/j.neucom.2019.11.029), Neurocomputing 2019.
- [Instance Search via Instance Level Segmentation and Feature Representation](https://arxiv.org/abs/1806.03576), arXiv 2018.
- [Unsupervised object discovery for instance recognition](https://doi.org/10.1109/WACV.2018.00194), WACV 2018.
- [Faster R-CNN Features for Instance Search](https://github.com/imatge-upc/retrieval-2016-deepvision), CVPR workshop 2016.

## ANN search

- [Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://arxiv.org/pdf/1908.10396.pdf), [blog](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html), [code](https://github.com/google-research/google-research/tree/master/scann), ICML 2020.
- [Improving Approximate Nearest Neighbor Search through Learned Adaptive Early Termination](https://www.pdl.cmu.edu/PDL-FTP/BigLearning/mod0246-liA.pdf), SIGMOD 2020.
- [RobustiQ A Robust ANN Search Method for Billion-scale Similarity Search on GPUs](http://users.monash.edu/~yli/assets/pdf/icmr19-sigconf.pdf), ICMR 2019.
- [Zoom: Multi-View Vector Search for Optimizing Accuracy, Latency and Memory](https://www.microsoft.com/en-us/research/uploads/prod/2018/08/zoom-multi-view-tech-report.pdf).
- [Vector and Line Quantization for Billion-scale Similarity Search on GPUs](http://users.monash.edu/~yli/assets/pdf/vlq_fgcs.pdf).
- [GGNN: Graph-based GPU Nearest Neighbor Search](https://github.com/cgtuebingen/ggnn), arxiv 2019, [code](https://github.com/cgtuebingen/ggnn).
- [Learning to Route in Similarity Graphs](https://arxiv.org/abs/1905.10987), ICML 2019.
- [Practical and Optimal LSH for Angular Distance](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=http%3A%2F%2Fpapers.nips.cc%2Fpaper%2F5893-practical-and-optimal-lsh-for-angular-distance.pdf).
- [pq-fast-scan](https://github.com/technicolor-research/pq-fast-scan).
- [faiss](https://github.com/facebookresearch/faiss). A library for efficient similarity search and clustering of dense vectors.
- [Polysemous codes](https://arxiv.org/abs/1609.01882).
- [Optimized Product Quantization](http://kaiminghe.com/cvpr13/index.html).
- [lopq](https://github.com/yahoo/lopq). Training of Locally Optimized Product Quantization (LOPQ) models for approximate nearest neighbor search of high dimensional data in Python and Spark.
- [nns_benchmark](https://github.com/DBWangGroupUNSW/nns_benchmark). Benchmark of Nearest Neighbor Search on High Dimensional Data.
- [Optimized Product Quantization](http://kaiminghe.com/cvpr13/index.html).
- [Falconn](https://github.com/FALCONN-LIB/FALCONN). FAst Lookups of Cosine and Other Nearest Neighbors.
- [Annoy](https://github.com/spotify/annoy). Approximate Nearest Neighbors in C++/Python optimized for memory usage and loading/saving to disk.
- [NMSLIB](https://github.com/searchivarius/nmslib). Non-Metric Space Library (NMSLIB): A similarity search library and a toolkit for evaluation of k-NN methods for generic non-metric spaces. 
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://github.com/nmslib/hnsw), graph-based method.
- [Fast Approximate Nearest Neighbor Search With Navigating Spreading-out Graphs](https://arxiv.org/abs/1707.00143), [code](https://github.com/ZJULearning/nsg)
- [Efficient Nearest Neighbors Search for Large-Scale Landmark Recognition](http://cn.arxiv.org/pdf/1806.05946.pdf)
- [NV-tree: A Scalable Disk-Based High-Dimensional Index](https://en.ru.is/media/skjol-td/PhDHerwig.pdf).
- [Dynamicity and Durability in Scalable Visual Instance Search](https://arxiv.org/abs/1805.10942).
- [Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors](https://arxiv.org/abs/1802.02422)，[code](https://github.com/dbaranchuk/ivf-hnsw).
- [Link and code: Fast indexing with graphs and compact regression codes](https://arxiv.org/abs/1804.09996).
- [A Survey of Product Quantization](https://www.jstage.jst.go.jp/article/mta/6/1/6_2/_pdf/)，对于矢量量化方法一篇比较完整的调研，值得一读.
- [GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints](https://arxiv.org/abs/1807.06294)，学习局部特征的descriptor，匹配能力较强.
- [Learning a Complete Image Indexing Pipeline](https://arxiv.org/pdf/1712.04480.pdf), CVPR 2018.
- [spreading vectors for similarity search](https://arxiv.org/abs/1806.03198), ICLR 2019.
- [SPTAG](urlhttps://github.com/microsoft/SPTAG): A library for fast approximate nearest neighbor search. Microsoft.

## CBIR Attack

- [Open Set Adversarial Examples](https://arxiv.org/abs/1809.02681).

## CBIR rank

- [Fast Spectral Ranking for Similarity Search](http://arxiv.org/pdf/1703.06935.pdf), [code](https://github.com/ducha-aiki/manifold-diffusion), CVPR 2018.

## CBIR in Industry

- [Videntifier](http://videntifier.com/) is a visual search engine based on a patented large-scale local feature database, [demo](http://flickrdemo.videntifier.com/), based on SIFT feature and NV-tree. ([Chinese blog post](https://yongyuan.name/blog/videntifier-and-nv-tree.html)).
- [Web-Scale Responsive Visual Search at Bing](https://arxiv.org/abs/1802.04914).
- [Visual Search at Alibaba](https://dl.acm.org/citation.cfm?id=3219819.3219820).
- [Visual Search at Pinterest](https://labs.pinterest.com/user/themes/pinlabs/assets/paper/visual_search_at_pinterest.pdf).
- [Visual Discovery at Pinterest](https://arxiv.org/abs/1702.04680).
- [Learning a Unified Embedding for Visual Search at Pinterest](https://arxiv.org/abs/1908.01707), KDD 2019.
- [Visual Search at ebay]().
- [Deep Learning based Large Scale Visual Recommendation and Search for E-Commerce](https://arxiv.org/abs/1703.02344), [project](https://github.com/flipkart-incubator/fk-visual-search).
- [微信「扫一扫识物」 的背后技术揭秘](https://mp.weixin.qq.com/s/fiUUkT7hyJwXmAGQ1kMcqQ).
- [揭秘微信「扫一扫」识物为什么这么快？](https://mp.weixin.qq.com/s/EBCcBWob_iFa51-gOVPYQA)

## CBIR Competition and Challenge

- [Google Landmark Retrieval Challenge](https://www.kaggle.com/c/landmark-retrieval-challenge), 2018.
- [Alibaba Large-scale Image Search Challenge](https://tianchi.aliyun.com/competition/introduction.htm?raceId=231510&_lang=en_US), 2015.
- [Pkbigdata image retrieval](http://www.pkbigdata.com/common/cmpt/%E5%9B%BE%E5%83%8F%E6%90%9C%E7%B4%A2%E7%AB%9E%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html), 2015.
- [Large-scale Landmark Retrieval/Recognition under a Noisy and Diverse Dataset](https://arxiv.org/pdf/1906.04087.pdf), [Landmark2019-1st-and-3rd-Place-Solution](https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution).

## CBIR for Duplicate(copy) detection

- [A Robust and Fast Video Copy Detection System Using Content-Based Fingerprinting](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=0ahUKEwiisbW0maXYAhXLOY8KHUw0AEsQFgg7MAI&url=https%3A%2F%2Fpdfs.semanticscholar.org%2F7b4f%2F68e227999da8ffc6dc9f7fd34da5ebaad09f.pdf&usg=AOvVaw0mZvcT7VhEuEm68oieXLv-).

## Feature Fusion

- [Feature fusion using Canonical Correlation Analysis](https://github.com/mhaghighat/ccaFuse).

## Instance Matching

- [AdaLAM: Revisiting Handcrafted Outlier Detection](https://arxiv.org/pdf/2006.04250.pdf), arxiv 2006.
- [Graph-Cut RANSAC](https://arxiv.org/abs/1706.00984), [code](https://github.com/danini/graph-cut-ransac)
- [Image Matching Benchmark](https://arxiv.org/pdf/1709.03917.pdf)
- [GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence](https://github.com/JiawangBian/GMS-Feature-Matcher)
- [A Vote-and-Verify Strategy for Fast Spatial Verification in Image Retrieval](https://github.com/vote-and-verify/vote-and-verify)
- [CODE: Coherence Based Decision Boundaries for Feature Correspondence]()
- [Robust feature matching in 2.3µs](https://www.edwardrosten.com/work/taylor_2009_robust.pdf)
- [PopSift is an implementation of the SIFT algorithm in CUDA](https://github.com/alicevision/popsift)
- [openMVG robust_estimation](https://github.com/openMVG/openMVG/tree/e3a0bde5e9c676d1cb663a38f7e74c771324d69a/src/openMVG/robust_estimation)
- [Neural-Guided RANSAC: Learning Where to Sample Model Hypotheses](https://arxiv.org/pdf/1905.04132v1.pdf).
- [Homography from two orientation- and scale-covariant features](https://arxiv.org/pdf/1906.11927.pdf), [code](https://github.com/danini/homography-from-sift-features).

## Semantic Matching

- [End-to-end weakly-supervised semantic alignment](https://github.com/ignacio-rocco/weakalign).

## Template Matching

- [QATM: Quality-Aware Template Matching For Deep Learning](https://arxiv.org/pdf/1903.07254.pdf), CVPR 2019.

## Image Identification

- [Image Identification Using SIFT Algorithm: Performance Analysis against Different Image Deformations](https://arxiv.org/pdf/1710.02728.pdf).

## Tutorials

- [PyRetri](https://github.com/PyRetri/PyRetri), Open source deep learning based image retrieval toolbox based on PyTorch.
- [How to Apply Distance Metric Learning to Street-to-Shop Problem](https://medium.com/mlreview/how-to-apply-distance-metric-learning-for-street-to-shop-problem-d21247723d2a).
- [Recent Image Search Techniques](http://cvpr2016.thecvf.com/program/tutorials).
- [Compact Features for Visual Search](http://cvpr2016.thecvf.com/program/tutorials).
- [multimedia-indexing](https://github.com/MKLab-ITI/multimedia-indexing). A framework for large-scale feature extraction, indexing and retrieval.
- [Image Similarity using Deep Ranking](https://medium.com/@akarshzingade/image-similarity-using-deep-ranking-c1bd83855978), [code](https://github.com/akarshzingade/image-similarity-deep-ranking).
- [Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss).
- [tf_retrieval_baseline](https://github.com/ahmdtaha/tf_retrieval_baseline).

## Slide

- [VRG Prague in “Large-Scale Landmark Recognition Challenge”](https://drive.google.com/file/d/1NFhfkqKjo_bXM-yuI3KbZt_iHRmiUyTG/view), ranked 3rd in the Google Landmark Recognition Challenge.

## Demo and Demo Online

- [Visual Image Retrieval and Localization](http://viral.image.ntua.gr/), SIFT feature encoded by BOW.
- [VGG Image Search Engine](https://gitlab.com/vgg/vise), SIFT feature encoded by BOW.
- [SoTu](https://github.com/zysite/SoTu), A flask-based cbir system.
- [yisou](https://yisou.yuanbin.me/), A flask-based painting cbir system, the search algorithm is designed by [Yong Yuan](http://yongyuan.name/).

## Datasets

- [DeepFashion2 Dataset](https://github.com/switchablenorms/DeepFashion2), DeepFashion2 is a comprehensive fashion dataset.
- [Holidays](https://rd.springer.com/chapter/10.1007/978-3-540-88682-2_24), Holidays consists images from personal holiday albums of various scene types.
- [Oxford](https://ieeexplore.ieee.org/document/4270197), Oxford consists of 11 different Oxford landmarks.
- [Paris](https://ieeexplore.ieee.org/abstract/document/4587635/), Paris consists of images crawled from 11 queries on specific Paris architecture.
- [ROxford and RParis](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html), ROxford and RParis are revisited versions of the original Oxford and Paris with annotation corrections, enlarged sizes and more difficult samples.
- [INSTRE](https://dl.acm.org/doi/abs/10.1145/2700292), INSTRE is an instance-level object retrieval dataset.
 
## Useful Package 

- [VLFeat](http://www.vlfeat.org/)
- [Yael](http://yael.gforge.inria.fr/)
