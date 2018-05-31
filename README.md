## Awesome image retrieval papers

### CVPR 2018

- [Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking](http://cmp.felk.cvut.cz/~toliageo/p/RadenovicIscenToliasAvrithisChum_CVPR2018_Revisiting%20Oxford%20and%20Paris:%20Large-Scale%20Image%20Retrieval%20Benchmarking.pdf), [project](http://cmp.felk.cvut.cz/revisitop/), CVPR 2018
- [Fast Spectral Ranking for Similarity Search](http://cn.arxiv.org/pdf/1703.06935.pdf), [code](https://github.com/ducha-aiki/manifold-diffusion), CVPR 2018
- [Learning a Complete Image Indexing Pipeline](https://arxiv.org/pdf/1712.04480.pdf), CVPR 2018

#### Local Feature Based

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

#### Deep Learning Feature (Global Feature)

- [Deep Image Retrieval:Learning Global Representations for Image earch](https://arxiv.org/abs/1604.01325)
- [End-to-end Learning of Deep Visual Representations for Image retrieval](), DIR更详细的论文说明
- [What Is the Best Practice for CNNs Applied to Visual Instance Retrieval?](), 关于layer选取的问题
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

#### Deep Learning Feature (Local Feature)

- [Learning Discriminative Affine Regions via Discriminability](http://cn.arxiv.org/pdf/1711.06704.pdf), [affnet](https://github.com/ducha-aiki/affnet)
- [A Large Dataset for Improving Patch Matching](http://cn.arxiv.org/pdf/1801.01466.pdf), [PS-Dataset](https://github.com/rmitra/PS-Dataset)
- [Working hard to know your neighbor's margins: Local descriptor learning loss](), [hardnet](https://github.com/DagnyT/hardnet)
- [MatchNet: Unifying Feature and Metric Learning for Patch-Based Matching](), [matchnet](https://github.com/hanxf/matchnet)

#### ANN search

- [Practical and Optimal LSH for Angular Distance](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=http%3A%2F%2Fpapers.nips.cc%2Fpaper%2F5893-practical-and-optimal-lsh-for-angular-distance.pdf)
- [pq-fast-scan](https://github.com/technicolor-research/pq-fast-scan)
- [faiss](https://github.com/facebookresearch/faiss). A library for efficient similarity search and clustering of dense vectors.
- [Polysemous codes]()
- [Optimized Product Quantization](http://kaiminghe.com/cvpr13/index.html)
- [lopq](https://github.com/yahoo/lopq). Training of Locally Optimized Product Quantization (LOPQ) models for approximate nearest neighbor search of high dimensional data in Python and Spark.
- [nns_benchmark](https://github.com/DBWangGroupUNSW/nns_benchmark). Benchmark of Nearest Neighbor Search on High Dimensional Data.
- [Optimized Product Quantization](http://kaiminghe.com/cvpr13/index.html)
- [Falconn](https://github.com/FALCONN-LIB/FALCONN). FAst Lookups of Cosine and Other Nearest Neighbors.
- [Annoy](https://github.com/spotify/annoy). Approximate Nearest Neighbors in C++/Python optimized for memory usage and loading/saving to disk 
- [NMSLIB](https://github.com/searchivarius/nmslib). Non-Metric Space Library (NMSLIB): A similarity search library and a toolkit for evaluation of k-NN methods for generic non-metric spaces. 
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://github.com/nmslib/hnsw), graph-based method.
- [Fast Approximate Nearest Neighbor Search With Navigating Spreading-out Graphs](https://arxiv.org/abs/1707.00143), [code](https://github.com/ZJULearning/nsg)

#### CBIR in Industry

- [Visual Search at Pinterest]()
- [Visual Discovery at Pinterest]()
- [Visual Search at ebay]()
- [Deep Learning based Large Scale Visual Recommendation and Search for E-Commerce](https://arxiv.org/abs/1703.02344), [project](https://github.com/flipkart-incubator/fk-visual-search)

#### CBIR Competition and Challenge

- [Google Landmark Retrieval Challenge](https://www.kaggle.com/c/landmark-retrieval-challenge), 2018
- [Alibaba Large-scale Image Search Challenge](https://tianchi.aliyun.com/competition/introduction.htm?raceId=231510&_lang=en_US), 2015
- [Pkbigdata image retrieval](http://www.pkbigdata.com/common/cmpt/%E5%9B%BE%E5%83%8F%E6%90%9C%E7%B4%A2%E7%AB%9E%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html]), 2015

#### CBIR for Duplicate(copy) detection

- [A Robust and Fast Video Copy Detection System Using Content-Based Fingerprinting](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=0ahUKEwiisbW0maXYAhXLOY8KHUw0AEsQFgg7MAI&url=https%3A%2F%2Fpdfs.semanticscholar.org%2F7b4f%2F68e227999da8ffc6dc9f7fd34da5ebaad09f.pdf&usg=AOvVaw0mZvcT7VhEuEm68oieXLv-)

#### Feature fusion

- [Feature fusion using Canonical Correlation Analysis](https://github.com/mhaghighat/ccaFuse)

#### Feature Matching

- [Image Matching Benchmark](https://arxiv.org/pdf/1709.03917.pdf)
- [GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence](https://github.com/JiawangBian/GMS-Feature-Matcher)
- [A Vote-and-Verify Strategy for Fast Spatial Verification in Image Retrieval](https://github.com/vote-and-verify/vote-and-verify)
- [CODE: Coherence Based Decision Boundaries for Feature Correspondence]()
- [Robust feature matching in 2.3µs](https://www.edwardrosten.com/work/taylor_2009_robust.pdf)
- [PopSift is an implementation of the SIFT algorithm in CUDA](https://github.com/alicevision/popsift)
- [openMVG robust_estimation](https://github.com/openMVG/openMVG/tree/e3a0bde5e9c676d1cb663a38f7e74c771324d69a/src/openMVG/robust_estimation)

#### Plan to read

- [VisualRank: Applying PageRank to Large-Scale Image Search]()

### Tutorials

- [Recent Image Search Techniques](http://cvpr2016.thecvf.com/program/tutorials)
- [Compact Features for Visual Search](http://cvpr2016.thecvf.com/program/tutorials)
- [multimedia-indexing](https://github.com/MKLab-ITI/multimedia-indexing). A framework for large-scale feature extraction, indexing and retrieval.
- [Image Similarity using Deep Ranking](https://medium.com/@akarshzingade/image-similarity-using-deep-ranking-c1bd83855978), [code](https://github.com/akarshzingade/image-similarity-deep-ranking).

### Demo online

- [Visual Image Retrieval and Localization](http://viral.image.ntua.gr/), sift feature encoded by BOW.

### Useful Package

- [VLFeat](http://www.vlfeat.org/)
- [Yael](http://yael.gforge.inria.fr/)
