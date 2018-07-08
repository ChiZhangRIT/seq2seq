## Sentence-to-sentence Model

This work is based on easy_seq2seq. Original code can be found [here](https://github.com/suriyadeepan/easy_seq2seq).

This code is re-written based on [sentence-to-sentence model](https://github.com/ChiZhangRIT/video2txt/tree/master/sent2sent) in Tensorflow v0.12.

### Data preparation

Some characters in the original data is not in the right codec. We are using our own dataset.

All sentences pairs were extracted from MSCOCO + Flickr30k + MSR-VTT + MSVD.

### Training

Edit *seq2seq.ini* file to set *mode = train*. To use pre-trained embedding, set *use_pretrained_embedding = true*
```
python execute.py
```
Note: Set *trainable=True* in *embedding = vs.get_variable(...)* (line 96) in *embedding/rnn_cell.py* to enable training on pre-trained embedding.
Note: To assgin a GPU device, use
```
export CUDA_VISIBLE_DEVICES="0"
```

### Evaluation

Edit *seq2seq.ini* file to set *mode = eval*
```
python execute.py
```

### Inference

Edit *seq2seq.ini* file to set *mode = test*.
Edit *seq2seq.ini* file to set *mode = generate* to generate paraphrasing sentences, given a file containing multiple input sentences.
```
python execute.py
```

### TensorBoard

Run a TensorBoard server in a separate process for real-time monitoring of training progress and evaluation metrics.
```
tensorboard --logdir=log/ --port=6364
```

### Contact

If you have any questions please contact Chi Zhang at cxz2081@rit.edu.