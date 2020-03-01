<div align="center">
	<img width="500" height="350" src="logo.svg" alt="Awesome">
	<br>
  <p>
    <a href="https://github.com/willard-yuan/awesome-cbir-papers">CBIR in academia and industry</a>
  </p>
</div>

# Awesome image retrieval papers

The main goal is collect classical and solid work of image retrieval in academia and industry.

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

- [ARXIV](#ARXIV)
- [Local Feature Based](#Local-Feature-Based)
- [Deep Learning Feature (Global Feature)](#Deep-Learning-Feature-(Global-Feature))
- [Deep Learning Feature (Local Feature)](#Deep-Learning-Feature-(Local-Feature))
- [ANN search](#ANN-search)
- [CBIR rank](#CBIR-rank)
- [CBIR in Industry](#CBIR-in-Industry)
- [CBIR Competition and Challenge](#CBIR-Competition-and-Challenge)
- [CBIR for Duplicate(copy) detection](#CBIR-for-Duplicate(copy)-detection)
- [Feature Fusion](#Feature-Fusion)
- [Instance Matching](#Instance-Matching)
- [Semantic Matching](#Semantic-Matching)
- [Image Identification](#Image-Identification)
- [Tutorials](#Tutorials)
- [Demo and Demo Online](#Demo-and-Demo-Online)
- [Datasets](#Datasets)
- [Useful Package](#Useful-Package)

## ARXIV

- [A Benchmark on Tricks for Large-scale Image Retrieval](https://arxiv.org/pdf/1907.11854.pdf)，通用图像检索各种trick介绍。
- [Learning with Average Precision: Training Image Retrieval with a Listwise Loss](https://arxiv.org/pdf/1906.07589v1.pdf), deep image retrieval续作。
- [MultiGrain: a unified image embedding for classes and instances](https://arxiv.org/abs/1902.05509)，Hervé Jégou, Andrea Vedaldi, Matthijs Douze，三牛首次出现在同一paper中。将相似和实例检索统一在一个框架中。
- [Composing Text and Image for Image Retrieval - An Empirical Odyssey](https://arxiv.org/pdf/1812.07119.pdf)
- [Visualizing Deep Similarity Networks](https://arxiv.org/pdf/1901.00536.pdf)
- [Combination of Multiple Global Descriptors for Image Retrieval](https://github.com/naver/cgd)

## Local Feature Based

- [Object retrieval with large vocabularies and fast spatial matching](https://www.robots.ox.ac.uk/~vgg/publications/papers/philbin07.pdf)
- [Visual Categorization with Bags of Keypoints](http://www.cs.princeton.edu/courses/archive/fall09/cos429/papers/csurka-eccv-04.pdf)
- [ORB: an efficient alternative to SIFT or SURF](https://www.willowgarage.com/sites/default/files/orb_final.pdf)
- [Object Recognition from Local Scale-Invariant Features](http://www.cs.ubc.ca/~lowe/papers/iccv99.pdf)
- [Total Recall: Automatic Query Expansion with a Generative Feature Model for Object Retrieval](https://www.robots.ox.ac.uk/~vgg/publications/papers/philbin07.pdf)
- [Three things everyone should know to improve object retrieval](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)
- [On-the-fly learning for visual search of large-scale image and video datasets](https://www.robots.ox.ac.uk/~vgg/publications/2015/Chatfield15/chatfield15.pdf)
- [All about VLAD]()
- [Aggregating localdescriptors into a compact image representatio]()
- [More About VLAD: A Leap from Euclidean to Riemannian Manifolds]()
- [Hamming embedding and weak geometric consistency for large scale image search]()
- [Revisiting the VLAD image representation](https://hal.inria.fr/hal-00840653v1/document), [project](https://github.com/jorjasso/VLAD/blob/master/VLADlib/VLAD.py)
- [Improving the Fisher Kernel for Large-Scale Image Classification](https://www.robots.ox.ac.uk/~vgg/rg/papers/peronnin_etal_ECCV10.pdf)
- [Image Classification with the Fisher Vector: Theory and Practice](https://hal.inria.fr/hal-00830491/document)
- [Democratic Diffusion Aggregation for ImageRetrieval]()
- [A Vote-and-Verify Strategy for Fast Spatial Verification in Image Retrieval]()
- [Triangulation embedding and democratic aggregation for image search]()
- [Efficient Large-scale Image Search With a Vocabulary Tree](http://www.ipol.im/pub/art/2018/199/), [code](https://github.com/fragofer/voctree)

## Deep Learning Feature (Global Feature)

- [SOLAR: Second-Order Loss and Attention for Image Retrieval](https://arxiv.org/abs/2001.08972v2), arxiv 2020.
- [Deep Image Retrieval:Learning Global Representations for Image search](https://arxiv.org/abs/1604.01325)
- [End-to-end Learning of Deep Visual Representations for Image retrieval](https://arxiv.org/abs/1610.07940), DIR更详细的论文说明
- [What Is the Best Practice for CNNs Applied to Visual Instance Retrieval?](https://arxiv.org/abs/1611.01640), 关于layer选取的问题
- [Bags of Local Convolutional Features for Scalable Instance Search](https://arxiv.org/abs/1604.01325)
- [Faster R-CNN Features for Instance Search](https://github.com/imatge-upc/retrieval-2016-deepvision)
- [Cross-dimensional Weighting for Aggregated Deep Convolutional Features](https://arxiv.org/abs/1512.04065), [project](https://github.com/yahoo/crow)
- [Class-Weighted Convolutional Features for Image Retrieval](https://github.com/imatge-upc/retrieval-2017-cam)
- [Multi-Scale Orderless Pooling of Deep Convolutional Activation Features](), VLAD coding
- [Aggregating Deep Convolutional Features for Image Retrieval](https://arxiv.org/abs/1510.07493), [论文笔记](https://zhuanlan.zhihu.com/p/23136747), [基于深度学习的视觉实例搜索研究进展](https://zhuanlan.zhihu.com/p/22265265).
- [Particular object retrieval with integral max-pooling of CNN activations](https://arxiv.org/abs/1511.05879), [project](http://cmp.felk.cvut.cz/~toliageo/soft.html)
- [Particular object retrieval using CNN](https://github.com/AaltoVision/Object-Retrieval)
- [Learning to Match Aerial Images with Deep Attentive Architectures](https://vision.cornell.edu/se3/wp-content/uploads/2016/04/1204.pdf).
- [Siamese Network of Deep Fisher-Vector Descriptors for Image Retrieval](https://arxiv.org/pdf/1702.00338v1.pdf)
- [Combining Fisher Vector and Convolutional Neural Networks for Image Retrieval](http://ceur-ws.org/Vol-1653/paper_19.pdf), fv和cnn特征融合提升
- [Selective Deep Convolutional Features for Image Retrieval](https://arxiv.org/pdf/1707.00809v1.pdf)
- [Class-Weighted Convolutional Features for Image Retrieval](https://github.com/imatge-upc/retrieval-2017-cam)
- [Towards Good Practices for Image Retrieval Based on CNN Features]()
- [Fine-tuning CNN Image Retrieval with No Human Annotation](https://arxiv.org/abs/1711.02512)
- [An accurate retrieval through R-MAC+ descriptors for landmark recognition](https://arxiv.org/pdf/1806.08565.pdf)
- [Regional Attention Based Deep Feature for Image Retrieval](https://sglab.kaist.ac.kr/RegionalAttention/), [code](https://github.com/jaeyoon1603/Retrieval-RegionalAttention), BMVC 2018.
- [Detect-to-Retrieve: Efficient Regional Aggregation for Image Search](https://arxiv.org/pdf/1812.01584.pdf), arxiv.
- [Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking](http://cmp.felk.cvut.cz/~toliageo/p/RadenovicIscenToliasAvrithisChum_CVPR2018_Revisiting%20Oxford%20and%20Paris:%20Large-Scale%20Image%20Retrieval%20Benchmarking.pdf), [project](http://cmp.felk.cvut.cz/revisitop/), CVPR 2018.
- [Guided Similarity Separation for Image Retrieval](https://github.com/layer6ai-labs/GSS), NeurIPS 2019.

## Deep Learning Feature (Local Feature)

- [UR2KiD: Unifying Retrieval, Keypoint Detection, and Keypoint Description without Local Correspondence Supervision](https://arxiv.org/abs/2001.07252), arxiv.
- [Beyond Cartesian Representations for Local Descriptors](https://arxiv.org/abs/1908.05547), [code](https://github.com/cvlab-epfl/log-polar-descriptors), ICCV 2019.
- [R2D2: Reliable and Repeatable Detector and Descriptor](https://arxiv.org/abs/1906.06195), [R2D2](https://github.com/naver/r2d2), NeurIPS 2019.
- [SOSNet: Second Order Similarity Regularization for Local Descriptor Learning](https://github.com/scape-research/SOSNet), CVPR 2019.
- [Local Features and Visual Words Emerge in Activations](https://avrithis.net/data/pub/pdf/conf/C110.cvpr19.spatial.pdf), CVPR 2019.
- [Explicit Spatial Encoding for Deep Local Descriptors](https://arxiv.org/abs/1904.07190), CVPR 2019.
- [Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters](https://github.com/axelBarroso/Key.Net), ICCV 2019.
- [Learning Discriminative Affine Regions via Discriminability](http://cn.arxiv.org/pdf/1711.06704.pdf), [affnet](https://github.com/ducha-aiki/affnet)
- [A Large Dataset for Improving Patch Matching](http://cn.arxiv.org/pdf/1801.01466.pdf), [PS-Dataset](https://github.com/rmitra/PS-Dataset)
- [Working hard to know your neighbor's margins: Local descriptor learning loss](), [hardnet](https://github.com/DagnyT/hardnet)
- [MatchNet: Unifying Feature and Metric Learning for Patch-Based Matching](), [matchnet](https://github.com/hanxf/matchnet)
- [LF-Net: Learning Local Features from Images](https://arxiv.org/abs/1805.09662), NeurIPS 2018.
- [Local Descriptors Optimized for Average Precision](http://openaccess.thecvf.com/content_cvpr_2018/papers/He_Local_Descriptors_Optimized_CVPR_2018_paper.pdf), CVPR 2018
- [SuperPoint: Self-Supervised Interest Point Detection and Description](http://cn.arxiv.org/pdf/1712.07629.pdf), Magic Leap
- [GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints](https://arxiv.org/pdf/1807.06294.pdf), [code](https://github.com/lzx551402/geodesc), ECCV 2018.
- [Learning local feature descriptors with triplets and shallow convolutional neural networks](https://github.com/vbalnt/tfeat), BMVC 2016.

## ANN search

- [RobustiQ A Robust ANN Search Method for Billion-scale Similarity Search on GPUs](http://users.monash.edu/~yli/assets/pdf/icmr19-sigconf.pdf), ICMR 2019.
- [Zoom: Multi-View Vector Search for Optimizing Accuracy, Latency and Memory](https://www.microsoft.com/en-us/research/uploads/prod/2018/08/zoom-multi-view-tech-report.pdf)
- [Vector and Line Quantization for Billion-scale Similarity Search on GPUs](http://users.monash.edu/~yli/assets/pdf/vlq_fgcs.pdf)
- [GGNN: Graph-based GPU Nearest Neighbor Search](https://github.com/cgtuebingen/ggnn), arxiv 2019.
- [Learning to Route in Similarity Graphs](https://arxiv.org/abs/1905.10987), ICML 2019.
- [Practical and Optimal LSH for Angular Distance](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=http%3A%2F%2Fpapers.nips.cc%2Fpaper%2F5893-practical-and-optimal-lsh-for-angular-distance.pdf)
- [pq-fast-scan](https://github.com/technicolor-research/pq-fast-scan)
- [faiss](https://github.com/facebookresearch/faiss). A library for efficient similarity search and clustering of dense vectors.
- [Polysemous codes](https://arxiv.org/abs/1609.01882)
- [Optimized Product Quantization](http://kaiminghe.com/cvpr13/index.html)
- [lopq](https://github.com/yahoo/lopq). Training of Locally Optimized Product Quantization (LOPQ) models for approximate nearest neighbor search of high dimensional data in Python and Spark.
- [nns_benchmark](https://github.com/DBWangGroupUNSW/nns_benchmark). Benchmark of Nearest Neighbor Search on High Dimensional Data.
- [Optimized Product Quantization](http://kaiminghe.com/cvpr13/index.html)
- [Falconn](https://github.com/FALCONN-LIB/FALCONN). FAst Lookups of Cosine and Other Nearest Neighbors.
- [Annoy](https://github.com/spotify/annoy). Approximate Nearest Neighbors in C++/Python optimized for memory usage and loading/saving to disk 
- [NMSLIB](https://github.com/searchivarius/nmslib). Non-Metric Space Library (NMSLIB): A similarity search library and a toolkit for evaluation of k-NN methods for generic non-metric spaces. 
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://github.com/nmslib/hnsw), graph-based method.
- [Fast Approximate Nearest Neighbor Search With Navigating Spreading-out Graphs](https://arxiv.org/abs/1707.00143), [code](https://github.com/ZJULearning/nsg)
- [Efficient Nearest Neighbors Search for Large-Scale Landmark Recognition](http://cn.arxiv.org/pdf/1806.05946.pdf)
- [NV-tree: A Scalable Disk-Based High-Dimensional Index](https://en.ru.is/media/skjol-td/PhDHerwig.pdf)
- [Dynamicity and Durability in Scalable Visual Instance Search](https://arxiv.org/abs/1805.10942)
- [Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors](https://arxiv.org/abs/1802.02422)，[code](https://github.com/dbaranchuk/ivf-hnsw)
- [Link and code: Fast indexing with graphs and compact regression codes](https://arxiv.org/abs/1804.09996)
- [A Survey of Product Quantization](https://www.jstage.jst.go.jp/article/mta/6/1/6_2/_pdf/)，对于矢量量化方法一篇比较完整的调研，值得一读
- [GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints](https://arxiv.org/abs/1807.06294)，学习局部特征的descriptor，匹配能力较强
- [Learning a Complete Image Indexing Pipeline](https://arxiv.org/pdf/1712.04480.pdf), CVPR 2018
- [spreading vectors for similarity search](https://arxiv.org/abs/1806.03198), ICLR 2019.
- [SPTAG](urlhttps://github.com/microsoft/SPTAG): A library for fast approximate nearest neighbor search. Microsoft.

## CBIR rank

- [Fast Spectral Ranking for Similarity Search](http://cn.arxiv.org/pdf/1703.06935.pdf), [code](https://github.com/ducha-aiki/manifold-diffusion), CVPR 2018

## CBIR in Industry

- [Videntifier](http://videntifier.com/) is a visual search engine based on a patented large-scale local feature database, [demo](http://flickrdemo.videntifier.com/), based on SIFT feature and NV-tree.
- [Web-Scale Responsive Visual Search at Bing](https://arxiv.org/abs/1802.04914)
- [Visual Search at Alibaba](https://dl.acm.org/citation.cfm?id=3219819.3219820)
- [Visual Search at Pinterest](https://labs.pinterest.com/user/themes/pinlabs/assets/paper/visual_search_at_pinterest.pdf)
- [Visual Discovery at Pinterest](https://arxiv.org/abs/1702.04680)
- [Learning a Unified Embedding for Visual Search at Pinterest](https://arxiv.org/abs/1908.01707), KDD 2019.
- [Visual Search at ebay]()
- [Deep Learning based Large Scale Visual Recommendation and Search for E-Commerce](https://arxiv.org/abs/1703.02344), [project](https://github.com/flipkart-incubator/fk-visual-search)
- [微信「扫一扫识物」 的背后技术揭秘](https://mp.weixin.qq.com/s/fiUUkT7hyJwXmAGQ1kMcqQ)

## CBIR Competition and Challenge

- [Google Landmark Retrieval Challenge](https://www.kaggle.com/c/landmark-retrieval-challenge), 2018
- [Alibaba Large-scale Image Search Challenge](https://tianchi.aliyun.com/competition/introduction.htm?raceId=231510&_lang=en_US), 2015
- [Pkbigdata image retrieval](http://www.pkbigdata.com/common/cmpt/%E5%9B%BE%E5%83%8F%E6%90%9C%E7%B4%A2%E7%AB%9E%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html), 2015
- [Large-scale Landmark Retrieval/Recognition under a Noisy and Diverse Dataset](https://arxiv.org/pdf/1906.04087.pdf), [Landmark2019-1st-and-3rd-Place-Solution](https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution).

## CBIR for Duplicate(copy) detection

- [A Robust and Fast Video Copy Detection System Using Content-Based Fingerprinting](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=0ahUKEwiisbW0maXYAhXLOY8KHUw0AEsQFgg7MAI&url=https%3A%2F%2Fpdfs.semanticscholar.org%2F7b4f%2F68e227999da8ffc6dc9f7fd34da5ebaad09f.pdf&usg=AOvVaw0mZvcT7VhEuEm68oieXLv-)

## Feature Fusion

- [Feature fusion using Canonical Correlation Analysis](https://github.com/mhaghighat/ccaFuse)

## Instance Matching

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

- [End-to-end weakly-supervised semantic alignment](https://github.com/ignacio-rocco/weakalign)

## Image Identification

- [Image Identification Using SIFT Algorithm: Performance Analysis against Different Image Deformations](https://arxiv.org/pdf/1710.02728.pdf)

## Tutorials

- [How to Apply Distance Metric Learning to Street-to-Shop Problem](https://medium.com/mlreview/how-to-apply-distance-metric-learning-for-street-to-shop-problem-d21247723d2a)
- [Recent Image Search Techniques](http://cvpr2016.thecvf.com/program/tutorials)
- [Compact Features for Visual Search](http://cvpr2016.thecvf.com/program/tutorials)
- [multimedia-indexing](https://github.com/MKLab-ITI/multimedia-indexing). A framework for large-scale feature extraction, indexing and retrieval.
- [Image Similarity using Deep Ranking](https://medium.com/@akarshzingade/image-similarity-using-deep-ranking-c1bd83855978), [code](https://github.com/akarshzingade/image-similarity-deep-ranking).
- [Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss)
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
 
## Useful Package 

- [VLFeat](http://www.vlfeat.org/)
- [Yael](http://yael.gforge.inria.fr/)
