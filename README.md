# AudioSet models + tools
- The repo provides PyTorch transcribed audioset classifiers, including VGGish and YAMNet, along with utilities to manipulate autioset category ontology tree.
- I personaly use it to annotate large amount of raw audio files with semantic labels. The code is pretty cleaned up. Should be usable off-the-shelf. 

## Background
Google open-sourced a few models trained on [AudioSet](https://research.google.com/audioset/). They first released VGGish, followed by [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet), 
which performs better and is more lightweight. 
If you search online you can find pytorch versions of VGGish, but not YAMNet. This repo is for that. 

In addition there is a pipeline for audio file labeling. 

## To convert weights
You may install the package in developer mode using 
```bash
pip install --editable .
```


To convert yamnet, put 'tf_2_torch/convert_yamnet.py' into the yamnet repository.
Download the tensorflow yamnet weight using
```bash
curl -O https://storage.googleapis.com/audioset/yamnet.h5
```
and then run the standalone [conversion utility](tf_2_torch/convert_yamnet.py)
```bash
python convert_yamnet.py
```

## To label audio files
Look at [tools/label_audio_data.py](tools/label_audio_data.py)
I am making minimal assumptions about what your data looks like. 
To label your audio dataset, create a dataloader yourself by replacing the placeholder dummy. 
The output will be saved in a json file of the schema:
~~~yaml
    {
        model: vggish,
        model_categories: [
            {
                name: specch,
                id: 3759347
            },
            ...
        ],
        predictions: [
            {
                id: 12345,
                category_tsr_fname: 12345.npy
                per_chunk_length: 0.96 (in seconds)
                meta: {} an optional payload copied verbatim from the inputs
            }
            ...
        ]
    }
~~~

## AudioSet ontology
The metadata for AudioSet is stored in [ontology.json](torch_audioset/audioset_ontology/ontology.json). It has been cleaned up substantially 
for ease of use.
~~~
The ontology json file format
id:                 /m/0dgw9r,
name:Male           speech, man speaking,
description:        A description of the class in a few lines.
citation_uri:       Any text used as the basis for the description. e.g. a Wikipedia page
positive_examples:  YouTube URLs of positive examples
child_ids:          ids of children classes of this class
restrictions:       ['abstract', 'blacklist'] a list of optional tags.
                    'abstract': the class is purely a parent class
                    'blacklist': the class is ambiguous to annotate
~~~

You can manipulate the ontology tree using the [provided tree](torch_audioset/audioset_ontology/ontology.py) data structure. 


## Input Audio Processing
An important component of audio processing is conversion to spectrogram. Every implementation uses a slightly varied version
of spectrogram generation, and it's a little confusing at times. In fact, the original YAMNet and VGGish are a little different
from each other. See [the tflow_input_processing modules](torch_audioset/data/tflow_input_processing)

My [pytorch version]((torch_audioset/data)) does work, even though the spectrograms cannot exactly match. The semantic predictions are fine.  

When a single audio file is too long (longer than 1 hour), you will likely see out of memory error on a 12GB mem card. Hence the model 
automatically chunks audio files into hourly segments to prevent the problem. But this runs under the hood. 

