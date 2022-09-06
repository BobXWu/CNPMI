# Cross-lingual Normalized Pointwise Mutual Information (CNPMI)

This is a repo of CNPMI that evalutes the coherence and alignment of cross-lingual topics.


## Usage

### 1. Prepare Environment

    scikit-learn==1.0.2
    pyyaml==6.0


### 2. Prepare Reference Corpus

We use [WikiComp](https://linguatools.org/tools/corpora/wikipedia-comparable-corpora/) to generate parallel documents as reference copora.
In `./ref_corpus`, we include the reference corpus for English & Chinese and English & Japanese.


### 3. Run Evaluation

Run the following command to compute the CNPMI of cross-lingual topics:

    python CNPMI.py \
        --topics1 {path to topics of languge 1} \
        --topics2 {path to topics of languge 2} \
        --ref_corpus_config ./configs/ref_corpus/{lang1_lang2}.yaml

## Reference

Our code is based on

[topic_interpretability](https://github.com/jhlau/topic_interpretability)

[Lessons from the Bible on Modern Topics:Low-Resource Multilingual Topic Model Evaluation](https://aclanthology.org/N18-1099.pdf)
