# Get results of evaluation

import argparse
import os

from collections import defaultdict
import numpy as np
import pandas as pd


ABBREVIATIONS_MAP ={'human':'Human Subject', 'nonhuman':'General Subject', #forgery semantic
                 'visibleimage':'RGB Images', 'nirimage':'Near-infrared Images', 'visibleimagetext':'RGB Images and Texts', 'video':'Videos', #forgery odality
                 'category':'Forgery Binary Classification', 'seg':'Forgery Spatial Localization (Segmentation)', 'det':'Forgery Spatial Localization (Detection)', 'temporal':'Forgery Temporal Localization', #forgery tasks
                 'entiresynthesis':'Entire Synthesis', 'faceantispoofing':'Spoofing', 'faceedit':'Face Editing', 'faceeditfacetransferoneface':'Face Editing & Face Transfer', 'faceedittextattributemanipulation':'Face Editing & Text Attribute Manipulation',
                 'faceedittextswap':'Face Editing & Text Swap', 'facereenactment':'Face Reenactment', 'faceswapmultiplefaces':'Face Swap (Multiple Faces)', 'faceswaponeface':'Face Swap (Single Face)', 'faceswaponefacefaceedit':'Face Swap (Single Face) & Face Editing', 'faceswaponefacetextattributemanipulation':'Face Swap (Single Face) & Text Attribute Manipulation',
                 'faceswaponefacetextswap':'Face Swap (Single Face) & Text Swap', 'facetransferoneface':'Face Transfer', 'styletransfer':'Style Translation', 'textattributemanipulation':'Text Attribute Manipulation', 'textswap':'Text Swap', 'generalcopymove':'Copy-Move', 'generalremove':'Removal', 
                 'generalsplicing':'Splicing', 'imageenhancement':'Image Enhancement', 'outofcontext':'Out-of-Context', 'real':'Real media without being forged', #forgery types
                 'diffusion':'Diffusion models', 'gan':'Generative Adversarial Networks', 'proprietary':'Proprietary', '3D':'3D masks', 'papercut':'Paper-Cut', 'print':'Print', 'replay':'Replay', 'encoderdecoder':'Encoder-Decoder', 'encoderdecodertraditional':'Encoder-Decoder&Graphics-based methods',
                 'encoderdecodertransformer':'Encoder-Decoder&Transformer', 'gantransformer':'Generative Adversarial Networks&Transformer', 'encoderdecoderretrieval':'Encoder-Decoder&Retrieval-based methods', 'ganretrieval':'Generative Adversarial Networks&Retrieval-based methods', 'RNN':'Recurrent Neural Networks', 'traditional':'Graphics-based methods',
                 'encoderdecoderRNNtraditional':'Encoder-Decoder&Recurrent Neural Networks&Graphics-based methods', 'unknown':'Unknown (in the wild)', 'vae':'Variational Auto-Encoders', 'transformer':'Transformer', 'retrieval':'Retrieval-based methods', 'autoregressive':'Auto-regressive models', 'decoder':'Decoder', 'none':'Real media without being forged' #forgery models
                 }
ASPECTS = ['forgery semantic', 'forgery modality', 'forgery task', 'forgery type', 'forgery model']


def wrap_load_data(data_dict, read_dim='forgery semantic'):
    if read_dim == 'forgery semantic':
        POSITION_FLAG = 0
    elif read_dim == 'forgery type':
        POSITION_FLAG = 1
    elif read_dim == 'forgery model':
        POSITION_FLAG = 2
    elif read_dim == 'forgery modality':
        POSITION_FLAG = 3
    elif read_dim == 'forgery task':
        POSITION_FLAG = 4
    
    category_list = list(data_dict.keys())
    merge_category_accuracy = defaultdict(list)
    
    for i in range(len(category_list)):
        category_split = category_list[i].split('_')
        merge_category_accuracy[ABBREVIATIONS_MAP[category_split[POSITION_FLAG]]].append(
            data_dict[category_list[i]]
            )

    for k, v in merge_category_accuracy.items():
        score = np.array(merge_category_accuracy[k]).mean()
        merge_category_accuracy[k] = score

    return merge_category_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument("--detail", action='store_true')
    args = parser.parse_args()

    # Load results
    df = pd.read_csv(args.filename)
    data_dict = df.iloc[0].to_dict()
    overall_score = data_dict['Overall']
    del data_dict['split'], data_dict['Overall']

    # Analyse results
    res = dict()
    for aspect in ASPECTS:
        scores_per_aspect = wrap_load_data(data_dict, read_dim=aspect)
        res[aspect] = scores_per_aspect
    
    # Summarize results
    print()
    print(f"Overall score: {overall_score:.1%}")
    print("==============")

    print("Aspects breakdown:")
    print()
    for k, v in res.items():
        print(f"{k}: {np.mean(list(v.values())):.1%}")
        if args.detail:
            print("=======")
            print("Including: ", end="\n")
            for k_, v_ in v.items():
                print(f"{k_}: {v_:.1%}", end="; ")
            print()
            print("=======")
            print()
    print()