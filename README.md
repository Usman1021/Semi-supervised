## Semi-Supervised learning for Face Anti-Spoofing using Apex frame
#### Authors: Usman Muhammad, Mourad Oussalah and Jorma Laaksonen


##

### Abstract
Conventional feature extraction techniques in the face anti-spoofing domain either analyze the entire video sequence or focus on a specific segment for representation. The uncertainty is due to the lack of consensus on the number of frames that contribute to performance improvement. In this paper, we address this issue by using Gaussian weighting to generate Apex frames for videos. Specifically, an Apex frame is derived from a video by computing a weighted sum of its frames, where the weights are determined using a Gaussian distribution centered around the video's central frame. Furthermore, we explore various temporal lengths to produce multiple unlabeled Apex frames using a Gaussian function, without the need for convolution. By doing so, we leverage the benefits of semi-supervised learning, which considers both labeled and unlabeled Apex frames to effectively discriminate between live and spoof classes. Our key contribution emphasizes the Apex frame's capacity to represent video content cohesively, while unlabeled Apex frames facilitate efficient semi-supervised learning, as they enable the model to learn from videos of varying temporal lengths. Experimental results using four face anti-spoofing databases: CASIA, REPLAY-ATTACK, OULU-NPU, and MSU-MFSD demonstrate the Apex frame's efficacy in advancing face anti-spoofing techniques.
