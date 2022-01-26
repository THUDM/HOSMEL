# HOSMEL

## Usage

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

To train your own module, we recommend to copy the relation module, ideally, for training you only need to make sure your training data satisfies the form of

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

Then reimplement the 

## <span id="ld">Live Demonstration</span>

We provided a live demonstration at `http://60.205.221.159/el/`

## Links to the model Checkpoints

```apl
https://drive.google.com/file/d/12w12GH5XEVGKYoaWm_sXVFHGFOSFJHnu/view?usp=sharing
https://drive.google.com/file/d/1BZphOj8rS7qHZA3wWz0vcY3H_qbCjTGK/view?usp=sharing
https://drive.google.com/file/d/1pMqN63yy9S9NZJWRV41bc-dASRndLwtr/view?usp=sharing
https://drive.google.com/file/d/1xKvPx0LY6XgVXY7wtSmUwk2iMfBm-9qw/view?usp=sharing
```

## <span id="su">Setting Up</span>

### dependencies

### Mention Filtering

### Mention Detection

### Disambiguation By Subtitle

### Disambiguation By Relation
