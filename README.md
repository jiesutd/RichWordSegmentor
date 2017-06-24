RichWordSegmentor
======
RichWordSegmentor is a package for Word Segmentation using transition based neural networks under LibN3L package. It is the state-of-the-art neural word segmentator which supports rich pretraining from external data. With the help of rich pretraining, our model achieves the best result on 5 out of 6 Chinese word segmentation benchmarks. Performance details and model structure can be seen in our ACL paper: [Neural word segmentation with rich pretraining](https://arxiv.org/abs/1704.08960). 

Demo system: (coming...)
======
* Download the [LibN3L](https://github.com/SUTDNLP/LibN3L) library and configure your system. Please refer to [Here](https://github.com/SUTDNLP/LibN3L)
* Open [CMakeLists.txt](CMakeLists.txt) and change " ../LibN3L/" into the directory of your [LibN3L](https://github.com/SUTDNLP/LibN3L) package.
* Run the [demo.sh](demo.sh) file: `sh demo.sh` (didn't load pretrained char/bichar embeddings in this demo script.)

The demo system includes Chinese word segmentation sample data ["train.debug"](example/train.debug), ["dev.debug"](example/dev.debug) and ["test.debug"](example/test.debug), Chinese word embeding sample file ["ctb.50d.word.debug"](example/ctb.50d.word.debug), Chinese char and char bigram pretrained embedding sample file ["char.emb"](example/char.emb) ["bichar.emb"](example/bichar.emb)and parameter setting file["option.STD"](example/option.STD). All of these files are gathered at folder [RichWordSegmentor/example](example).

Run: (to be filled)
=======
`cmake .`  
`make`

Training model:  
`./STDSeg -l -train ${train.data} -dev ${dev.data} -test ${dev.data} -option ${option.file} -model ${save_model_to_file} -word ${pretrain_word_emb, optional} -char ${pretrain_char_emb, optional} -bichar ${pretrain_bichar_emb, optional} -numlayer ${pretrain_parameters, optional}`

Load model:  
`./STDSeg -test ${test.data} -model ${load_model_file} -output ${output_file}`

Input:
======
1. For evaluate model performance, word seperated by a space, each sentence take one line. For example:

   就 做 了 一点 微小 的 工作 ， 谢谢 大家 。  
   一个人 的 命运 啊 ， 当然 要 靠 自我 奋斗 ， 但是 也要 考虑 到 历史 的 行程 。

   Result will calculate the P/R/F automatically.

2. For raw text decoding, one sentence each line (without space).   

   就做了一点微小的工作，谢谢大家。  
   一个人的命运啊，当然要靠自我奋斗，但是也要考虑到历史的行程。

Output:
=======
The same format with training data. Word seperated by a space, each sentence take one line.

就 做 了 一点 微小 的 工作 ， 谢谢 大家 。  
一个人 的 命运 啊 ， 当然 要 靠 自我 奋斗 ， 但是 也要 考虑 到 历史 的 行程 。

Trained model/embeddings/parameters of rich pretraining and baseline:
==========
We shared our trained model at [BaiduPan](https://pan.baidu.com/s/1pLO6T9D)(https://pan.baidu.com/s/1pLO6T9D) for visiters reproducing our results.  
1. File `ctb.bilstm.joint4.model`: 
   the trained model on CTB6.0 corpus using multitask pretraining. You can simply load this file to decode raw text without training. Run:  

   `./STDSeg -test ${input_raw_text} -model ctb.bilstm.joint4.model -output ${output_segmentated_text}`

2. File `joint4.all.b10c1.2h.iter17.mchar, .mbichar, .pmodel` are pretrained character, character bigram embeddings and representing parameters.
   If you want to train your own model, you can load these three files following above instruction.   

3. File: `gigaword_chn.all.a2b.uni.ite50.vec`, `gigaword_chn.all.a2b.bi.ite50.vec` and `ctb.50d.vec` are the char, bichar and word embeddings of our baseline, respectively.

4. If you want to do the rich pretraining experiments (for generating three files in last item), please refer to [TrainEmbMultiTask](https://github.com/jiesutd/TrainEmbMultiTask).  


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


Note: 
======
* Current version only compatible with [LibN3L](https://github.com/SUTDNLP/LibN3L) after ***Dec. 10th 2015*** , which contains the model saving and loading module.
* The example files are just to verify the running for the code. For copyright consideration, we take only hundreds of sentences as example. Hence the results on those example datasets does not represent the real performance on large dataset.



Cite: 
========
    @article{yang2017neural,  
     title={Neural Word Segmentation with Rich Pretraining},  
     author={Yang, Jie and Zhang, Yue and Dong, Fei},  
     journal={arXiv preprint arXiv:1704.08960},  
     year={2017}  
    }  

Update
====
* 2017-April-4: init version


