# Awesome AI and Datascience:
This list is a personal collection of papers, books, articles, guides, and courses that I’ve read (or started reading) to learn and grow in the field of AI. It’s essentially a record of the resources that helped me understand key concepts, sharpen my skills, and progress in my career. I created it as a way to keep track of what’s been useful—whether for revisiting topics I’ve forgotten, preparing for interviews, or simply staying organized. Over time, this list will evolve alongside me, reflecting my journey and growth in AI. I hope it also serves as a resource for others who are learning or following a similar path. A roadmap may be added in the future to outline my learning journey in a more structured way.
- [Awesome AI and Datascience](#awesome-ai-and-datascience)
  - [AI and Games](#ai-and-games)
  - [Computer Vision](#computer-vision)
    - [Image Classification](#image-classification)
    - [Fine-Grained Image Classification](#fine-grained-image-classification)
    - [Object Detection](#object-detection)
    - [Segmentation](#segmentation)
    - [Explainability](#explainability)
    - [Edge Detection](#edge-detection)
    - [Spatial Pyramid Pooling](#spatial-pyramid-pooling)
  - [Data Engineering](#data-engineering)
  - [Deep Learning](#deep-learning)
    - [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
    - [Multi-Task Learning](#multi-task-learning)
    - [Recurrent Neural Networks (RNN)](#recurrent-neural-networks-rnn)
    - [Gated Recurrent Unit (GRU)](#gated-recurrent-unit-gru)
  - [Edge Computing](#edge-computing)
    - [Machine Learning Compilers and Optimizers](#machine-learning-compilers-and-optimizers)
    - [TensorRT](#tensorrt)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Principal component analysis, Correspondence Analysis, Multiple Correspondence Analysis](#principal-component-analysis-correspondence-analysis-multiple-correspondence-analysis)
  - [Fundamentals](#fundamentals)
    - [Docker](#docker)
    - [Git](#git)
    - [Web Scraping](#web-scraping)
  - [Foundation Models](#foundation-models)
    - [Multimodal](#multimodal)
  - [Generative Models](#generative-models)
    - [Variational Autoencoders](#variational-autoencoders)
    - [Diffusion Models](#diffusion-models)
  - [Graph Neural Networks](#graph-neural-networks)
  - [Information Retrieval](#information-retrieval)
    - [Semantic Search](#semantic-search)
  - [Machine Learning](#machine-learning)
    - [Boosting Algorithms](#boosting-algorithms)
    - [Support Vector Machines](#support-vector-machines)
    - [Preparing for a Machine Learning Interview](#preparing-for-a-machine-learning-interview)
  - [Mathematics](#mathematics)
    - [Optimization for Machine Learning](#optimization-for-machine-learning)
  - [Natural Language Processing](#natural-language-processing)
    - [Transformers](#transformers)
  - [Python](#python)
    - [Clean Code](#clean-code)
    - [Data Structures](#data-structures)
    - [Distribution](#distribution)
    - [Documentation](#documentation)
    - [Efficient Code](#efficient-code)
    - [Generators](#generators)
    - [Metaclasses](#metaclasses)
    - [Tools and Development Environments](#tools-and-development-environments)
    - [Preparing for a Coding Interview](#preparing-for-a-coding-interview)
  - [Pytorch](#pytorch)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Statistics](#statistics)
    - [Estimation, Confidence Interval, Hypothesis Testing](#estimation-confidence-interval-hypothesis-testing)
    - [Monte Carlo Method](#monte-carlo-method)
  - [Interesting Reads](#interesting-reads)
# AI and Games:
- ## Books:
    - [Intelligence artificielle, une approche ludique - Tristan Cazenave](https://www.scribd.com/document/478586440/Intelligence-Artificielle-une-Approche-Ludique-pdf)
- ## Monte Carlo Search:
    - ### Blogs, Tutorials and Courses:
        - [Monte Carlo Search - Tristan Cazenave](https://www.lamsade.dauphine.fr/~cazenave/MonteCarlo.pdf)
# Computer Vision:
- ## Papers:
    - [Gradient-Based Learning Applied to Document
Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- ## Blogs, Tutorials and Courses:
    - [The dumb reason your fancy Computer Vision app isn’t working: Exif Orientation](https://medium.com/@ageitgey/the-dumb-reason-your-fancy-computer-vision-app-isnt-working-exif-orientation-73166c7d39da)
- ## Books:
    - [Computer Vision:
Algorithms and Applications
2nd Edition - Richard Szeliski](https://szeliski.org/Book/)
- ## Image Classification:
    - ### Papers:
        - [ImageNet Classification with Deep Convolutional
Neural Networks (AlexNet Paper)](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
        - [Deep Residual Learning for Image Recognition (ResNet Paper)](https://arxiv.org/pdf/1512.03385)
        - [Going Deeper with Convolutions (Inception Paper)](https://arxiv.org/pdf/1409.4842)
        - [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (MobileNet Paper)](https://arxiv.org/pdf/1704.04861)
        - [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG Paper)](https://arxiv.org/abs/1409.1556)
        - [Xception: Deep Learning with Depthwise Separable Convolutions (Xception Paper)](https://arxiv.org/pdf/1610.02357)
        - [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187)
        - [Compounding the Performance Improvements of Assembled Techniques in a Convolutional Neural Network](https://arxiv.org/pdf/2001.06268)
        - [ResNet strikes back: An improved training procedure in timm](https://arxiv.org/pdf/2110.00476)
    - ### Blogs, Tutorials and Courses:
        - [An Intuitive Guide to Deep Network Architectures - Towards Data Science](https://towardsdatascience.com/an-intuitive-guide-to-deep-network-architectures-65fdc477db41)
        - [How to Train State-Of-The-Art Models Using TorchVision’s Latest Primitives](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/)
    - ### Books:
        - [Deep Learning Bible - 2. Classification](https://wikidocs.net/book/7972)
- ## Fine-Grained Image Classification:
    - ### Papers:
        - [Fine-Grained Image Analysis with Deep Learning: A Survey](https://arxiv.org/abs/2111.06119)
- ## Object Detection:
    - ### Papers:
        - [Rich feature hierarchies for accurate object detection and semantic segmentation
Tech report (R-CNN Paper)](https://arxiv.org/pdf/1311.2524)
        - [Fast R-CNN](https://arxiv.org/pdf/1504.08083)
        - [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
        - [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325v5)
        - [You Only Look Once:
Unified, Real-Time Object Detection (The first yolo paper)](https://arxiv.org/pdf/1506.02640)
        - [YOLO9000:
Better, Faster, Stronger (YOLOv2)](https://arxiv.org/pdf/1612.08242)
        - [YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767)
        - [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934)
        - [YOLOv6: A Single-Stage Object Detection Framework for Industrial
Applications](https://arxiv.org/pdf/2209.02976)
        - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object
detectors](https://arxiv.org/pdf/2207.02696)
        - [YOLOv9: Learning What You Want to Learn
Using Programmable Gradient Information](https://arxiv.org/pdf/2402.13616)
        - [YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/pdf/2405.14458)
        - [YOLOV11: AN OVERVIEW OF THE KEY ARCHITECTURAL
ENHANCEMENTS](https://arxiv.org/pdf/2410.17725)
        - [Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630)
        - [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/pdf/1911.08287)
    - ### Blogs, Tutorials and Courses:
        - [yolov8_in_depth](https://github.com/akashAD98/yolov8_in_depth?tab=readme-ov-file)
        - [YOLOv11: One Concept You Must Know in Object Detection — Letterbox](https://medium.com/@gavin_xyw/letterbox-in-object-detection-77ee14e5ac46)
        - [What is YOLOv5? A Guide for Beginners.](https://blog.roboflow.com/yolov5-improvements-and-evaluation/)
    - ### Books:
        - [Deep Learning Bible - 4. Object Detection](https://wikidocs.net/book/8119)
- ## Segmentation:
    - ### Papers:
        - [Mask R-CNN](https://arxiv.org/pdf/1703.06870)
        - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
        - [nnU-Net: Self-adapting Framework
for U-Net-Based Medical Image Segmentation](https://arxiv.org/pdf/1809.10486)
        - [A survey of loss functions for semantic
segmentation](https://arxiv.org/pdf/2006.14822)
- ## Explainability:
    - ### Papers:
        - [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
- ## Edge Detection:
    - ### Papers:
        - [Quality Assessment Methods to Evaluate the Performance of Edge Detection Algorithms for Digital Image: A Systematic Literature Review](https://ieeexplore.ieee.org/document/9454489)
    - ### Blogs, Tutorials and Courses:
        - [Canny Edge Detection Step by Step in Python](https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123)
- ## Spatial Pyramid Pooling:
    - ### Papers:
        - [Spatial Pyramid Pooling in Deep Convolutional
Networks for Visual Recognition](https://arxiv.org/pdf/1406.4729v4)
# Data Engineering:
- ## Books:
    - [Data-Intensive Text Processing
with MapReduce](http://www.iro.umontreal.ca/~nie/IFT6255/Books/MapReduce.pdf)
    - [Hadoop: The Definitive Guide](https://www.isical.ac.in/~acmsc/WBDA2015/slides/hg/Oreilly.Hadoop.The.Definitive.Guide.3rd.Edition.Jan.2012.pdf)
# Deep Learning:
- ## Blogs, Tutorials and Courses:
    - [Deep Learning Specialization - Andrew NG](https://www.coursera.org/specializations/deep-learning)
    - [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/?fbclid=IwAR14qzU0WPypUSd2cJDn8_3GVDh6VjIcHBHcVJsLN9t7HtUkUfxzrluaaYY)
    - [GroupNorm,BatchNorm,InstanceNorm,LayerNorm](https://medium.com/@zljdanceholic/groupnorm-then-batchnorm-instancenorm-layernorm-e2b2a1d350a0)
- ## Long Short-Term Memory (LSTM):
    - ### Papers:
        - [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)
    - ### Blogs, Tutorials and Courses:
        - [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
        - [Illustrated Guide to LSTM’s and GRU’s: A step by step explanation](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
- ## Multi-Task Learning:
    - ### Papers:
        - [Multi-Task Learning Using Uncertainty to Weigh Losses
for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115)
    - ### Blogs, Tutorials and Courses:
        - [
Deep Multi-Task Learning — 3 Lessons Learned](https://towardsdatascience.com/deep-multi-task-learning-3-lessons-learned-7d0193d71fd6)
        - [Multi-Task Learning with Pytorch and FastAI](https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855)
        - [Multi Task Learning with Homoscedastic Uncertainty Implementation](https://github.com/ranandalon/mtl)
- ## Recurrent Neural Networks (RNN):
    - ### Blogs, Tutorials and Courses:
        - [Illustrated Guide to Recurrent Neural Networks](https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9)
- ## Gated Recurrent Unit (GRU):
    - ### Papers:
        - [Learning Phrase Representations using RNN Encoder–Decoder
for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078)
# Edge Computing:
- ## Machine Learning Compilers, optimizers and Gpu accelerated deep learning:
    - ### Blogs, Tutorials and Courses:
        - [A friendly introduction to machine learning compilers and optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)
- ## TensorRT:
    - ### Blogs, Tutorials and Courses:
        - [Profiling Deep Learning Networks And Automatic Mixed Precision For Optimization](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
# Exploratory Data Analysis:
- ## Principal component analysis, Correspondence Analysis, Multiple Correspondence Analysis:
    - ### Blogs, Tutorials and Courses:
        - [Analyse des données - Patrice Bertrand et Denis Pasquignon  (French Course)](https://www.ceremade.dauphine.fr/~pasquignon/analyse-des-donnees-M1.pdf)
# Fundamentals:
- ## Docker:
    - ### Blogs, Tutorials and Courses:
        - [Simplified guide to using Docker for local development environment](https://blog.atulr.com/docker-local-environment/)
        - [Use the same Dockerfile for both local development and production with multi-stage builds](https://blog.atulr.com/docker-local-production-image/)
        - [Introduction to docker - Datacamp](https://www.datacamp.com/courses/introduction-to-docker)
- ## Git:
    - ### Blogs, Tutorials and Courses:
        - [Introduction to Git for Data Science](https://www.datacamp.com/courses/introduction-to-git)
- ## Web Scraping:
    - ### Blogs, Tutorials and Courses:
        - [Web Scraping with Python](https://www.datacamp.com/courses/web-scraping-with-python)
# Foundation Models:
- ## Multimodal:
    - ### Papers:
        - [Multimodal Neurons in Artificial Neural Networks](https://distill.pub/2021/multimodal-neurons/)
# Generative Models:
- ## Variational Autoencoders:
    - ### Blogs, Tutorials and Courses:
        - [The theory behind Latent Variable Models: formulating a Variational Autoencoder](https://theaisummer.com/latent-variable-models/#variational-autoencoders)
        - [How to Generate Images using Autoencoders](https://theaisummer.com/Autoencoder/)
        - [MIT 6.S191 Introduction to Deep Learning Lecture 4: Deep Generative Modeling](http://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L4.pdf)
- ## Diffusion Models:
    - ### Papers:
        - [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)
        - [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
        - [
Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
        - [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)
        - [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/pdf/2403.18103)

    - ### Blogs, Tutorials and Courses:
        - [DiffusionFastForward - Diffusion Theory](https://github.com/mikonvergence/DiffusionFastForward/blob/master/notes/01-Diffusion-Theory.md)
        - [How diffusion models work: the math from scratch](https://theaisummer.com/diffusion-models/?fbclid=IwAR1BIeNHqa3NtC8SL0sKXHATHklJYphNH-8IGNoO3xZhSKM_GYcvrrQgB0o)
        - [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
        - [ What are Diffusion Models? ](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#reverse-diffusion-process)
        - [Diffusion models from scratch in PyTorch - Deep Findr](https://www.youtube.com/watch?v=a4Yfz2FxXiY)
# Graph Neural Networks:
- ## Blogs, Tutorials and Courses:
    - [Machine Learning with Graphs - Stanford University](http://web.stanford.edu/class/cs224w/index.html#content)
    - [LiteratureDL4Graph, A comprehensive collection of recent papers on graph deep learning](https://github.com/DeepGraphLearning/LiteratureDL4Graph)
# Information Retrieval:
- ## Books:
    - [Introduction to Information Retrieval - Cambridge University, Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze](https://nlp.stanford.edu/IR-book/information-retrieval-book.html)
- ## Semantic Search:
    - ### Blogs, Tutorials and Courses:
        - [Semantic Search - Sentence Transformers Documentation](https://www.sbert.net/examples/applications/semantic-search/README.html) - A guide on using sbert for semantic search
# Machine Learning:
- ## Books:
    - [Python Machine Learning, 3rd Edition - Sebastian Raschka , Vahid Mirjalili](https://sebastianraschka.com/books/#python-machine-learning-3rd-edition)
- ## Blogs, Tutorials and Courses:
    - [Machine Learning - Stanford University, Andrew NG](https://www.coursera.org/learn/machine-learning)
- ## Boosting Algorithms:
    - ### Blogs, Tutorials and Courses:
        - [Boosting algorithms explained](https://towardsdatascience.com/boosting-algorithms-explained-d38f56ef3f30)
        - [A Gentle Introduction to the Gradient Boosting Algorithm for Machine Learning](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)
- ## Preparing for a machine learning interview:
    - ### Books:
        - [Introduction to Machine Learning Interviews](https://huyenchip.com/ml-interviews-book/)
    - ### Machine Learning System Design:
        - #### Books:
            - [Designing Machine Learning   Systems: An Iterative Process for Production-Ready Applications - Chip Huyen](https://www.amazon.com/Designing-Machine-Learning-Systems-Production-Ready/dp/1098107969)
- ## Support Vector Machines:
    - [An Introduction to Support Vector Machine (SVM) in Python](https://builtin.com/machine-learning/support-vector-machine)
# Mathematics:
  - ## Optimization for Machine Learning:
    - ### Blogs, Tutorials and Courses:
        - [Optimization for Machine Learning - Clément Royer](https://www.lamsade.dauphine.fr/%7Ecroyer/ensdocs/OAA/PolyOAA.pdf)
# Natural Language Processing:
- ## Transformers:
    - ### Papers:
        - [Attention is all you need](https://arxiv.org/abs/1706.03762)
        - [Large Language Models: A Survey](https://arxiv.org/pdf/2402.06196v2)
    - ### Blogs, Tutorials and Courses:
        - [Hugging Face Course](https://huggingface.co/course/chapter1/1)
        - [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
        - [Positional Encoding in Transformer](https://medium.com/@sachinsoni600517/positional-encoding-in-transformer-2cc4ec703076)
        - [Sinusoidal Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
# Python:
- ## Books:
    - [Learning Python: Powerful Object-Oriented Programming - Mark Lutz](https://www.amazon.com/Learning-Python-Powerful-Object-Oriented-Programming-ebook/dp/B00DDZPC9S/ref=as_li_ss_tl?crid=2OZ9IA8BKEZKO&dchild=1&keywords=python+programming+language&qid=1588165545&sprefix=python+progra,aps,315&sr=8-3&linkCode=sl1&tag=adilet-20&linkId=bbe99ec15d04e43dd44966d937725cad&language=en_US)
    - [Python Cookbook](https://www.amazon.com.be/Python-Cookbook-Alex-Martelli/dp/0596001673)
- ## Clean Code:
    -  ### Blogs, Tutorials and Courses:
        - [How to write beautiful python code with PEP 8](https://realpython.com/python-pep8/)  
        - [5 Different Meanings of Underscore in Python](https://towardsdatascience.com/5-different-meanings-of-underscore-in-python-3fafa6cd0379)
        - [f-Strings: A New and Improved Way to Format Strings in Python](https://realpython.com/python-f-strings/#f-strings-a-new-and-improved-way-to-format-strings-in-python)
        - [The Zen of Python](https://zen-of-python.info/)
        - [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
        - [Python Formatter Beautifier](https://codebeautify.org/python-formatter-beautifier)
- ## Data Structures:
    - ### Blogs, Tutorials and Courses:
        - [Binary Tree](https://emre.me/data-structures/binary-tree/)
        - [Binary Search Tree](https://emre.me/data-structures/binary-search-trees/)
        - [Queue in Python](https://www.geeksforgeeks.org/queue-in-python/)
        - [Sets in Python](https://realpython.com/python-sets/)
- ## Distribution:
    - ### Blogs, Tutorials and Courses:
        - [How to create a Python library](https://medium.com/analytics-vidhya/how-to-create-a-python-library-7d5aea80cc3f)
        - [How to upload your python package to PyPi](https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56)
- ## Documentation:
    - ### Blogs, Tutorials and Courses:
        - [Python Docstrings](https://www.datacamp.com/community/tutorials/docstrings-python)
        - [Auto-documenting a python project using Sphinx](https://betterprogramming.pub/auto-documenting-a-python-project-using-sphinx-8878f9ddc6e9)
- ## Efficient Code:
    - ### Blogs, Tutorials and Courses:
        - [Code Profiling](https://towardsdatascience.com/a-quick-and-easy-guide-to-code-profiling-in-python-58c0ed7e602b)
        - [Python's Counter: The Pythonic Way to Count Objects](https://realpython.com/python-counter/)
        - [Unpacking in Python](https://stackabuse.com/unpacking-in-python-beyond-parallel-assignment/)
        - [Using the Python zip() Function for Parallel Iteration](https://realpython.com/python-zip-function/)
- ## Generators:
    - ### Blogs, Tutorials and Courses:
        - [Python yield, Generators and Generator Expressions](https://www.programiz.com/python-programming/generator?fbclid=IwAR2aGKoKpvMJy2R-jINBQMM7qcOeBczS192k1c2TZYwiV7YAKc1XKhboS4k)
        - [How to Use Generators and yield in Python](https://realpython.com/introduction-to-python-generators/)
- ## Metaclasses:
- ### Blogs, Tutorials and Courses:
    - [Python Metaclasses](https://www.godaddy.com/engineering/2018/12/20/python-metaclasses/)
- ## Tools and development environements: 
  - ### Jupyter Notebook
    - #### Blogs, Tutorials and Courses:
        - [7 essential tips for writing with jupyter notebook](https://towardsdatascience.com/7-essential-tips-for-writing-with-jupyter-notebook-60972a1a8901)
- ## Preparing for a coding interview:
    - ### Books:
        - [Competitive Programmer’s Handbook - Antti Laaksonen (2018)](https://cses.fi/book/book.pdf)
        - [Introduction to Algorithms, 3rd Edition](https://www.amazon.com/dp/0262033844/ref=as_li_ss_tl?ie=UTF8&linkCode=sl1&tag=adilet-20&linkId=925f749322dc9e4485887dce6cbc8248&language=en_US)
    - ### Blogs, Tutorials and Courses:
        - [CP-Algorithms.com](https://cp-algorithms.com/)
        - [Grokking LeetCode: A Smarter Way to Prepare for Coding Interviews](https://medium.com/interviewnoodle/grokking-leetcode-a-smarter-way-to-prepare-for-coding-interviews-e86d5c9fe4e1)
        - [Interview School](https://interviews.school)
        - [Pramp - Mock interviews with peers](https://www.pramp.com/#/)
    - ### Algorithms:
        - ### Blogs, Tutorials and Courses:
            - [Dynamic Programming](https://emre.me/algorithms/dynamic-programming/) 
            - [Greedy Algorithms](https://emre.me/algorithms/greedy-algorithms/)
    - ### Python implementation for different coding problem patterns:
      - ### Blogs, Tutorials and Courses:
        - [Coding Patterns: In-place Reversal of a Linked List](https://emre.me/coding-patterns/in-place-reversal-of-a-linked-list/)
        - [4 types of tree traversal algorithms (Java implementation)](https://towardsdatascience.com/4-types-of-tree-traversal-algorithms-d56328450846)
        - [Coding Patterns: Depth First Search (DFS)](https://emre.me/coding-patterns/depth-first-search/)
        - [Coding Patterns: Breadth First Search (BFS)](https://emre.me/coding-patterns/breadth-first-search/)
# Pytorch:
- ## Blogs, Tutorials and Courses:
    - [Pytorch 101: An applied tutorial](https://www.youtube.com/watch?v=_R-mvKBD5U8&list=PL98nY_tJQXZln8spB5uTZdKN08mYGkOf2&index=1)
    - [PyTorch Training Performance Guide](https://residentmario.github.io/pytorch-training-performance-guide/intro.html)
# Reinforcement Learning:
- ## Books:
    - [Reinforcement Learning An Introduction - Richard S.Sutton, Andrew G. Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- ## Blogs, Tutorials and Courses:
    - [Reinforcement Learning - Stéphane Airiau](https://www.lamsade.dauphine.fr/~airiau/Teaching/M2-IASDapp-RL/) (French Course)
# Statistics:
- ## Estimation, Confidence Interval, Hypothesis Testing:
  - ### Blogs, Tutorials and Courses:
    - [Statistique mathématique - Vincent Rivoirard](https://www.ceremade.dauphine.fr/~rivoirar/Poly-L3-StatMath.pdf) (French Course)
    - [The Bootstrap Method for Standard Errors and Confidence Intervals](https://www.dummies.com/article/academics-the-arts/science/biology/the-bootstrap-method-for-standard-errors-and-confidence-intervals-164614/)
- ## Monte Carlo Method:
  - ### Books:
    - [Méthodes de Monte Carlo - Julien Stoehr](https://www.ceremade.dauphine.fr/~stoehr/data/medias/001/m1_monte_carlo/cm_monte_carlo.pdf) (French Course)
# Interesting Reads:
- [Do not end the week with nothing - Patio11](https://training.kalzumeus.com/newsletters/archive/do-not-end-the-week-with-nothing)
- [Learn in public - swyx](https://www.swyx.io/learn-in-public)
- [You Are Not Too Old (To Pivot Into AI)](https://www.latent.space/p/not-old)
- [The 2025 AI Engineer Reading List](https://www.latent.space/p/2025-papers)
- [History of Deep Learning](https://github.com/adam-maj/deep-learning?tab=readme-ov-file)
- [Tips for Writing Technical Papers ](https://cs.stanford.edu/people/widom/paper-writing.html)
- [A Survival Guide to a PhD](http://karpathy.github.io/2016/09/07/phd/?ref=ruder.io)
- [The Turing Way handbook](https://book.the-turing-way.org/)