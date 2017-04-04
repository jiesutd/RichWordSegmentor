RichWordSegmentor
======
RichWordSegmentor is a package for Word Segmentation using transition based neural networks under LibN3L package. It is the state-of-the-art neural word segmentator which supports rich pretraining from external data. With the help of rich pretraining, our model achieves the best result on 5 out of 6 Chinese word segmentation benchmarks. Performance details and model structure can be seen in our paper (insert paper link). 

Demo system (coming...)
======
* Download the [LibN3L](https://github.com/SUTDNLP/LibN3L) library and configure your system. Please refer to [Here](https://github.com/SUTDNLP/LibN3L)
* Open [CMakeLists.txt](CMakeLists.txt) and change " ../LibN3L/" into the directory of your [LibN3L](https://github.com/SUTDNLP/LibN3L) package.
* Run the [demo-entity.sh](demo-entity.sh) file: `sh demo-entity.sh`

The demo system includes Chinese word segmentation sample data ["ctb.sample.train"](example/ctb.sample.train), ["ctb.sample.dev"](example/ctb.sample.dev) and ["ctb.sample.test"](example/ctb.sample.test), Chinese word embeding sample file ["ctb.emb"](example/ctb.emb), Chinese char and char bigram pretrained embedding sample file ["char.emb"](example/char.emb) ["bichar.emb"](example/bichar.emb)and parameter setting file(["demo.option"](example/demo.option). All of these files are gathered at folder [RichWordSegmentor/example](example).
 

Note: 
======
* Current version only compatible with [LibN3L](https://github.com/SUTDNLP/LibN3L) after ***Dec. 10th 2015*** , which contains the model saving and loading module.
* The example files are just to verify the running for the code. For copyright consideration, we take only hundreds of sentences as example. Hence the results on those example datasets does not represent the real performance on large dataset.

Feature format
======

Input:
======
Word seperated by a space, each sentence take one line. For example:
就 做 了 一点 微小 的 工作 ， 谢谢 大家 。
一个人 的 命运 啊 ， 当然 要 靠 自我 奋斗 ， 但是 也要 考虑 到 历史 的 行程 。

Output:
=======
The sampe format with training data.

Trained model (coming...):
==========
We shared our trained model at pan.baidu.com for visiters reproducing our results.

Monitoring information
=====
During the running of this NER system, it may print out the follow log information:


`Iter 13 finished. Total time taken is: 1260.37s`

`dev:`

`Recall: P=57508/59929=0.959602, Accuracy:       P=57508/59723=0.962912, Fmeasure:       0.961254`

`Decode dev finished. Total time taken is: 96.299s`

`test:`

`Recall: P=77895/81579=0.954841, Accuracy:       P=77895/81159=0.959783, Fmeasure:       0.957306`

`Decode test finished. Total time taken is: 128.9s`

`Exceeds best previous performance of 0.960922. Saving model file..`

The first "Recall..." line shows the performance of the dev set and the second "Recall..." line shows 
you the performance of the test set.


Cite: (coming)
========


Updating...
====
* 2017-April-4: init version


