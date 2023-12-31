# myStopword
Stopword extraction for Burmese.

## Project Overview

In this study, we present a comprehensive exploration of stopword extraction techniques tailored for the Burmese language. Utilizing a manually segmented corpus of 212,836 Burmese sentences, we benchmark traditional methods such as term frequency and entropy against our novel proposals, word2vec_frequency and fasttext_frequency. Our findings showcase that these innovative embeddingfrequency approaches not only align with the established behavior of Zipf’s law but also offer promising avenues for enhancing text analysis in Burmese and potentially other under-resourced languages.

## Top 100 Stopwords with Word2Vec-Freq Approach

```
ပါ	0.5001643738267268
က	0.43338324079538554
တယ်	0.40791037396204854
ကို	0.3985872108921257
မ	0.32352669811486157
သည်	0.28025497208245026
နေ	0.27852310116294865
တာ	0.27293261609619107
တွေ	0.2530480728774235
ရ	0.24845093301845017
မှာ	0.24160164337408604
တဲ့	0.18462732301395635
များ	0.18272116385653722
တော့	0.17465001909181718
ဖြစ်	0.17116645936360972
ပြီး	0.1562622578992229
နဲ့	0.15475188730209172
ရှိ	0.14922130744169845
လို့	0.14046497582996584
တို့	0.13486325685934963
လည်း	0.12846340348117322
တစ်	0.1277017872145718
ခဲ့	0.12466646772890598
ဘူး	0.12236801885266628
သူ	0.1212537133715681
ပဲ	0.12097543346268412
ဆို	0.11651405732365054
လိုက်	0.11392749075154325
မှ	0.10808205529523571
လာ	0.10637328832469514
ပေး	0.10570825101217653
၏	0.10275774729894353
ကြ	0.10194145592234634
နိုင်	0.09634713962455708
သွား	0.0960908866030329
ပြော	0.09203961621431574
လား	0.08755726861110656
ရင်	0.08659004897622465
သော	0.08428593269958283
ထား	0.08234904167102614
မယ်	0.081068444127762
တွင်	0.07381250057399452
နှင့်	0.07242719684614403
လေး	0.06951099184949538
မှု	0.06882807209499806
ရေး	0.06742865781151246
လေ	0.06656132359014388
ရဲ့	0.06648374294461562
ပြီ	0.06532569770223036
လုပ်	0.06441316766849536
ချစ်	0.06415000021127104
လဲ	0.061773593203829844
ချင်	0.06025071545017098
ဒီ	0.0582098612849824
သိ	0.05780695851658628
ကောင်း	0.05694586815064166
နှစ်	0.056800764939343606
ဘာ	0.056193195858194
ပါစေ	0.055639367990544045
လို	0.05544146913939555
ဟုတ်	0.055187331670047544
အားပေး	0.05384078506333398
ဟာ	0.05381638615402966
အရမ်း	0.05359635154532164
နော်	0.05318874882612109
မြန်မာ	0.05233350992865751
ဖို့	0.04932663923338697
အတွက်	0.048244913628310356
ကြီး	0.04766301956546602
ထဲ	0.04764070096987594
ခြင်း	0.04700296087626859
ခု	0.04694735329579975
ကျွန်တော်	0.0468949040359635
ပြန်	0.046671296477650705
၍	0.045715973212853714
သို့	0.043253637316276614
ယောက်	0.04043126187651992
နိုင်ငံ	0.04010894920605036
ကြည့်	0.03990142268580259
လူ	0.03911696023817256
ပေါ့	0.03826007053468603
အောင်	0.03634381470355174
သူ့	0.03587995844668203
ဦး	0.03522456027637168
ရောက်	0.03473672052909056
သေး	0.03281650332494508
မည်	0.03248254994965918
မင်း	0.03246972513164846
လှ	0.03189885069942773
ကလေး	0.03175985867624956
သာ	0.03173001535140674
ရေ	0.031339392516231676
စေ	0.03120849945180369
ဘယ်	0.030878526178368833
မင်္ဂလာ	0.03082534434726852
ရာ	0.030504640080651492
သာဓု	0.030321855525558862
ဟု	0.029525614722707053
မိ	0.029328798422847906
စရာ	0.02929144772310091
```

## Top 100 Stopwords with FastText-Freq Approach

```
ပါ	0.5705483555793762
က	0.5555157480105523
ကို	0.4966003161430984
တယ်	0.452409020914442
မ	0.3988926144186187
နေ	0.3391181661841258
တာ	0.32561061250584905
သည်	0.30859745427735
မှာ	0.3084148747738067
တွေ	0.2939161159818294
ရ	0.2862799907409759
တော့	0.22349408472669416
ဖြစ်	0.20988354249446822
တဲ့	0.20414133081073685
များ	0.192016062064364
ပြီး	0.18867079555169305
လည်း	0.1756473006579001
နဲ့	0.16740331289944954
လို့	0.16662771104553603
တို့	0.1615140428826474
ရှိ	0.1597501506531668
ဆို	0.14898291128660812
ပဲ	0.14868729570310985
သူ	0.14393879678224472
ခဲ့	0.14000112309453483
ဘူး	0.1364063914574605
တစ်	0.13543136903908817
လိုက်	0.13502465238153297
မှ	0.1273058712800564
ကြ	0.11650021225011889
လာ	0.11613811891023551
သွား	0.11168504885051894
ပြော	0.10716512583351265
ပေး	0.10607480464917615
၏	0.10401720465057376
လား	0.10016697632313519
ရင်	0.09243391263264913
နိုင်	0.08860534054318948
ထား	0.08544214524778196
သော	0.08533093404951274
လေ	0.08182126927777915
မယ်	0.07945801446188724
လေး	0.07353052747532916
ပြီ	0.07024927375001251
ချင်	0.07024163543780809
လုပ်	0.06911905961666016
တွင်	0.06667369922222433
ချစ်	0.06532082588540332
လဲ	0.06367973887678545
နှင့်	0.06350921462530824
ဟုတ်	0.061267586720979615
သိ	0.06057178967824527
ရဲ့	0.05929251754620295
နော်	0.05927344159275151
ဘာ	0.05923623116207466
ရေး	0.058921300989491714
ဒီ	0.058642132319024466
ကောင်း	0.058506836174087244
မှု	0.0578902798245705
လို	0.056945525666463416
အရမ်း	0.0562755303147506
ဟာ	0.0533858645634212
အားပေး	0.051910986853298265
ပြန်	0.0515672207553551
နှစ်	0.05088856981147493
ပါစေ	0.049765597410909114
ကြီး	0.04777152126371252
ကျွန်တော်	0.046916031150974785
ခု	0.046883984495954156
ဖို့	0.04679069518621163
ထဲ	0.04551752181903311
အတွက်	0.045176144050681495
မြန်မာ	0.04472707870053909
ပေါ့	0.044429933264857724
၍	0.041910210557441464
ခြင်း	0.04129731202646479
သူ့	0.040358899310279535
ကြည့်	0.03954029845401927
လူ	0.03940368593732106
အောင်	0.03938146230755405
ယောက်	0.038043896911131185
သာ	0.035752393771924094
သေး	0.03538853480085882
နိုင်ငံ	0.0350894326729799
မင်း	0.03457535494433306
သို့	0.03336506652754349
ကလေး	0.03152793864846166
ဦး	0.03068379826829239
ရောက်	0.03038993911797788
ရေ	0.030159606021245837
တတ်	0.030108004293910962
စရာ	0.0300932681004436
ရယ်	0.02962801745493696
လှ	0.028865005092715897
ဘယ်	0.02825101748167741
ရာ	0.028245879635421314
ဟု	0.028159527979475248
ကြိုက်	0.027053360118930364
မည်	0.026764023663486985
တွေ့	0.026555315580575146
```

## File Information

1. [run_exp4.sh](https://github.com/ye-kyaw-thu/myStopWord/blob/main/exp4_100/run_exp4.sh)  
   The shell script I used for experiment number 4, which extracts only 100 stopwords.)
2. [extract_stopword.py](https://github.com/ye-kyaw-thu/myStopWord/blob/main/exp4_100/extract_stopword.py)  
   Python code for extracting Burmese stopwords using various approaches
3. [zipfs_fit.py](https://github.com/ye-kyaw-thu/myStopWord/blob/main/exp4_100/zipfs_fit.py)  
   Python code for generating graphs
4. [stopword_talk.pdf](https://github.com/ye-kyaw-thu/myStopWord/blob/main/slide/stopword_talk.pdf)  
   Presentation slide that I used at the iSAI-NLP 2023 conference.
5. [pos_tagger.py](https://github.com/ye-kyaw-thu/myStopWord/blob/main/exp4_100/postag/pos_tagger.py)  
   Python code for creating a POS-tag dictionary using the myPOS ver.3 corpus and performing rough POS tagging on extracted Burmese stopwords.  
6. [extract_highest_uniq_pos_ver2.py](https://github.com/ye-kyaw-thu/myStopWord/blob/main/exp4_100/postag_freq/extract_highest_uniq_pos_ver2.py)  
   Python code for extracting the unique highest-ranked POS tags from the input stopword file.  
7. [zipfs_graph4all.py](https://github.com/ye-kyaw-thu/myStopWord/blob/main/exp4_100/graph/zipfs_graph4all.py)   
   Python code for drawing Zipf's Law graphs.  
8. [exp4_100/stopwords*.txt](https://github.com/ye-kyaw-thu/myStopWord/tree/main/exp4_100)  
   These files contain the top 100 Burmese stopwords extracted using both baseline and our two newly proposed methods.
9. [exp4_100/graph/*.png](https://github.com/ye-kyaw-thu/myStopWord/tree/main/exp4_100/graph)   
   These PNG files contain graphs comparing the extracted stopwords with Zipf’s law. Some of these graphs are plotted using linear scales, while others use logarithmic scales.  

## Command-line Options 

```
$ python ./extract_stopword.py --help
usage: extract_stopword.py [-h]
                           [--methods {tf,normalised_tf,idf,normalised_idf,tf_idf,term_based_random_sampling,pmi,entropy,variance,cooccurrence_network,lm,word2vec,word2vec_frequency,fasttext,fasttext_frequency} [{tf,normalised_tf,idf,normalised_idf,tf_idf,term_based_random_sampling,pmi,entropy,variance,cooccurrence_network,lm,word2vec,word2vec_frequency,fasttext,fasttext_frequency} ...]]
                           [--frequency] [-n NUMBER]
                           corpus_filename output_filename

Retrieve the stopword list from a given corpus.

positional arguments:
  corpus_filename       Path to the input corpus file.
  output_filename       Path to the output stopword list file.

optional arguments:
  -h, --help            show this help message and exit
  --methods {tf,normalised_tf,idf,normalised_idf,tf_idf,term_based_random_sampling,pmi,entropy,variance,cooccurrence_network,lm,word2vec,word2vec_frequency,fasttext,fasttext_frequency} [{tf,normalised_tf,idf,normalised_idf,tf_idf,term_based_random_sampling,pmi,entropy,variance,cooccurrence_network,lm,word2vec,word2vec_frequency,fasttext,fasttext_frequency} ...]
                        Methods to use for stopword retrieval. Can be one or multiple
                        for hybrid approach.
  --frequency           If set, outputs the stopword followed by its frequency (or
                        method-based value).
  -n NUMBER, --number NUMBER
                        Number of stopwords to extract. Default is 100.
```

Before using this Python script, make sure you have the necessary libraries installed on your system, including argparse, collections, math (built-in), random (built-in), networkx, gensim, fasttext, and numpy. You can easily install these libraries using `pip install`, which is the recommended method to ensure all dependencies are met. 

## Note

Please be aware that in this repository, we have uploaded the source codes (Python and shell scripts), extracted stopword files, various note files, and graphs. However, we are not sharing our in-house corpus, which was used for this experiment. We are currently working on cleaning it up, and we plan to release the entire corpus once the process is complete.

## Citation

If you intend to utilize any code snippets or utilize the extracted stopwords in your research, we kindly request that you acknowledge and cite the following paper: 

Ye Kyaw Thu and Thepchai Supnithi, "Embedding Meets Frequency: Novel Approaches to Stopword Identification in Burmese", In Proceedings of the 18th International Joint Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP 2023), Nov 27 to 29, 2023, Bangkok, Thailand, pp. xx-xx [to appear]  
