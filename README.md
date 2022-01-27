# **HOSMEL: A Hot Swappable Modulized Entity Linking Toolkit for Chinese**

## Usage

First,

```bash
git clone git@github.com:THUDM/HOSMEL.git
```

Our toolkit allows 3 different levels of usages

### Ready-to-Use

If you are not interested to change our default setup, follow the steps of [Mention Filtering](#mf), [Mention Detection](#md), [Disambiguation By Subtitle](#ds), and [Disambiguation By Relation](#dr). A [Live demonstration](#ld) using the same structure is also available.

### Partial

We know some users might prefer to design their own entity disambiguation framework or has other needs, some level of high quality candidate entity retrieval is still required. As a result, we support partial usage. To do this, make sure you complete the setups(which is only downloading and extracting zip files), simply import the corresponding part of the toolkit, and use it in your preferred manner. For better illustration a sample usage for the complete pipeline could be found in 

````apl
https://drive.google.com/drive/folders/1eh-dJnKWJulPuZGsORii4fPW-zCmWS5k?usp=sharing
````

See [setups](#su) for more information.

### Easy-to-Change

To train your own module, we recommend to copy the `NewMudule` module, a template Module we created, ideally, for training you only need to make sure your training data satisfies the form of

```json
{
    "sentence": "The input text", 
    "Label": int(k) # Label id showing targetk is the correct value, 
    "mention": "the mention of the entity", # Note: for mention detection, leave the mention empty and make the targets as your candidate mentions
    "target0": "A", # The four candidate values 
    "target1": "B",
    "target2": "C",
    "target3": "D"
}
```

Then reimplement the `generatePair` method in the `apply{feature}.py` file for infer.

## <span id="ld">Live Demonstration</span>

We provided a live demonstration at http://60.205.221.159/el/

## Links to the model Checkpoints

```apl
https://drive.google.com/file/d/12w12GH5XEVGKYoaWm_sXVFHGFOSFJHnu/view?usp=sharing
https://drive.google.com/file/d/1BZphOj8rS7qHZA3wWz0vcY3H_qbCjTGK/view?usp=sharing
https://drive.google.com/file/d/1pMqN63yy9S9NZJWRV41bc-dASRndLwtr/view?usp=sharing
https://drive.google.com/file/d/1xKvPx0LY6XgVXY7wtSmUwk2iMfBm-9qw/view?usp=sharing
```

## <span id="su">Setting Up</span>

### dependencies

Our method requires a few python based dependencies:

```bash
pip install flask torch tqdm pickle json pyahocorasick datasets
```

Make sure you have all the dependencies installed to access all of our methods.

### Mention Filtering <span id="mf"/>

First Download [TriMention.zip](https://drive.google.com/file/d/12w12GH5XEVGKYoaWm_sXVFHGFOSFJHnu/view?usp=sharing) to the `TriMention` directory, then simply extract the zip packages. You should see your directory to look like,

```bash
TriMention/
├── bdi2relation.pkl
├── mention.py
├── nameTri
├── subList.json
└── web.py
```

The `TriMention` folder not only includes the basic Trie tree, it also comes with subtitle and relationship data which would be used for later sections. We separate the data-loading and processing for the consideration of better development experience since loading such data takes a large amount of time. To load the datas simply run

```bash
python mention.py
```

### Mention Detection <span id="md"/>

Download the [MD_checkpoint.zip](https://drive.google.com/file/d/1pMqN63yy9S9NZJWRV41bc-dASRndLwtr/view?usp=sharing) file and extract it to the `MCMention/model` folder. It should look like

```bash
MCMention/
├── applyMention.py
├── model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── preprocessData.py
└── train.py
```

### Disambiguation By Subtitle <span id="ds"/>

Download the [SD_checkpoint.zip](https://drive.google.com/file/d/1BZphOj8rS7qHZA3wWz0vcY3H_qbCjTGK/view?usp=sharing) to the `model` directory under `MCSubtitle`. Then unzip it. The final directory should look like

```bash
MCSubtitle/
├── applySubtitle.py
├── model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── preprocessData.py
└── train.py
```

### Disambiguation By Relation <span id="dr"/>

Download the [RD_checkpoint.zip](https://drive.google.com/file/d/1xKvPx0LY6XgVXY7wtSmUwk2iMfBm-9qw/view?usp=sharing) to `MCRelation/model/`, and unzip to get

```bash
MCRelation/
├── applyRelation.py
├── model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── preprocessData.py
└── train.py
```

## Launching HOSMEL

To launch the complete HOSMEL, we provide a flask based backend, simply run it as

```bash
python backend.py
```

## Data Release

We release our training data [here](https://drive.google.com/file/d/17s6j2i93LDOyzCAGf82PNGKOHRIaoM2x/view?usp=sharing)

Our test data is also available [here](https://drive.google.com/file/d/1A1ktpFtLGKGnZwFmVjsCWnVXzIPBDpMo/view?usp=sharing)

## Training

To train a new module, simply move the training data to corresponding folder and use

```bash
python preprocessData.py
```

Make sure you have the name right, for example the name for training data in the MCSubtitle filder is `subtitleData.json`. This should give a `processedData.json` file in the same directory. Then use 

```bash
python train.py
```

The model's checkpoint should be saved in the `model` folder.

## Usage after training new Module

Idealy, if you have selected your checkpoint and replaced the `model` folder with it, you don't need to change anything other than editing the `generatePairs` method. However, just in case, if you are interested to change model directory. In the `apply*.py` folder, change

```bash
model_location = os.path.join(os.path.dirname(__file__),"model")
```

into

```bash
model_location = "New checkpoint location"
```

Will do it.

To use the new module for infer, it is required to reimplement the `generatePairs` method. The generate Pair method takes the input `entity`, aka, the output of the previous module, and retrieves a list of "mention|attribute value" pairs. A `bdi_list` variable, containing the same amount of items as the pairs list with the `i'th` item being the `id` of the `i'th` pair's entity, is required to add the scores back to the corresponding entity.

Now to test your newly implemented module, import the `topkNew` method and use

```python
from TriMention.web import parse_mentions as mentionFiltering
from ... import ... as DisambiguationBy...
......
from NewModule.applyNew import topkNew as DisambiguationByNew
text = "A test text"
entities = mentionFiltering(text)
entities = DisambiguationBy...(text,entities,K=3)
......
entities = DisambiguationByNew(text,entities,K=3)
print(entities[0])
```

