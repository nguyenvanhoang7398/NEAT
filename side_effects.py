from utils import *
from constants import *
import spacy
import itertools
from collections import Counter
from tqdm import tqdm
import numpy as np
from nltk import sent_tokenize


from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import json
from keras.preprocessing.sequence import pad_sequences
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.merge import Concatenate
from keras.layers import Dense, Input, Lambda, merge, dot, Subtract
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import Dropout, Activation, Reshape
from keras.models import Model
import keras.backend as K


nlp_spacy = spacy.load("en_core_web_sm")
ADPS = {"and", "or", "of", "in", "the", "during", "on", "at", "from", "for"}


se_map = {
    "trouble": "difficult",
    "sluggishness": "sluggish",
    "difficulty": "difficult",
    "painful": "pain",
    "redness": "red",
    "decreased": "decrease",
    "increased": "increase",
    "lack": "loss",
    "shakiness": "shaking",
    "racing": "fast",
    "quick": "fast",
    "unpleasant": "bad",
    "soreness": "sore",
    "reddening": "red",
    "excess": "excessive",
    "unable": "difficult",
    "swollen": "swell",
    "swelling": "swell",
    "puffiness": "swell",
    "puffy": "swell",
    "twitching": "twitch",
    "bleeding": "bleed",
    "coughing": "cough",
    "numbness": "numb",
    "itching": "itch",
    "vomiting": "vomit",
    "paleness": "pale",
    "urination": "urine",
    "absence": "loss",
    "bloody": "bleed",
    "blood": "bleed",
    "irritating": "irritate",
    "irritation": "irritate",
    "irritated": "irritate",
    "weakness": "weak",
    "breathing": "breath",
    "shortness": "short",
    "tightness": "tight",
    "warmth": "warm",
    "heat": "hot"
}


seed = 10
np.random.seed(seed)

embed_dim = 300

HIDDEN_DIM = 128
NUM_EPOCHS = 10
BATCH_SIZE = 16

lmtzr = WordNetLemmatizer()


def lemmatize(word):
    lemmatized_token = lmtzr.lemmatize(word)
    if lemmatized_token in se_map:
        return se_map[lemmatized_token]
    return lemmatized_token


def init_embedding_weights(i2w, w2vmodel):
    # Create initial embedding weights matrix
    # Return: np.array with dim [vocabsize, embeddingsize]

    d = 300
    V = len(i2w)
    assert sorted(i2w.keys()) == list(range(V))  # verify indices are sequential

    emb = np.zeros([V,d])
    num_unknownwords = 0
    unknow_words = []
    for i,l in i2w.items():
        if i==0:
            continue
        if l in w2vmodel.vocab:
            emb[i, :] = w2vmodel[l]
        else:
            num_unknownwords += 1
            unknow_words.append(l)
            emb[i] = np.random.uniform(-1, 1, d)
    return emb, num_unknownwords, unknow_words


def build_model(_maxlen, _maxlen_char_word, _idx2word, _w2v, _vocsize, _charsize, _embed_dim, _char_embed_dim,
                nclasses, _embeds):
    hiddendim = HIDDEN_DIM
    main_input = Input(shape=[_maxlen], dtype='int32', name='input') # (None, 36)
    char_input = Input(shape=[_maxlen, _maxlen_char_word], dtype='int32', name='char_input') # (None, 36, 25)

    embed = Embedding(input_dim=_vocsize, output_dim=_embed_dim, input_length=_maxlen,
                      weights=[_embeds], mask_zero=False, name='embedding', trainable=False)(main_input)

    embed = Dropout(0.5, name='embed_dropout')(embed)

    char_embed =  Embedding(input_dim=_charsize, output_dim=_char_embed_dim, embeddings_initializer='lecun_uniform',
                            input_length=_maxlen_char_word, mask_zero=False, name='char_embedding')(char_input)

    s = char_embed.shape
    char_embed = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], _char_embed_dim)))(char_embed)

    fwd_state = GRU(150, return_state=True)(char_embed)[-2]
    bwd_state = GRU(150, return_state=True, go_backwards=True)(char_embed)[-2]
    char_embed = Concatenate(axis=-1)([fwd_state, bwd_state])
    char_embed = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * 150]))(char_embed)
    char_embed = Dropout(0.5, name='char_embed_dropout')(char_embed)

    W_embed = Dense(300, name='Wembed')(embed)
    W_char_embed = Dense(300, name='W_charembed')(char_embed)
    merged1 = merge([W_embed, W_char_embed], name='merged1', mode='sum')
    tanh = Activation('tanh', name='tanh')(merged1)
    W_tanh = Dense(300, name='w_tanh')(tanh)
    a = Activation('sigmoid', name='sigmoid')(W_tanh)

    t = Lambda(lambda x: K.ones_like(x, dtype='float32'))(a)

    merged2 = merge([a, embed], name='merged2', mode='mul')
    sub = Subtract()([t, a])
    merged3 = merge([sub, char_embed], name='merged3', mode='mul')
    x_wave = merge([merged2, merged3], name='final_re', mode='sum')

    auxc = Dense(nclasses, name='auxiliary_classifier')(x_wave)
    auxc = Activation('softmax')(auxc) # (None, 36, 5) # (None, 36, 5)

    bi_gru = Bidirectional(GRU(hiddendim, return_sequences=True, name='gru'), merge_mode='concat', name='bigru')(x_wave) # (None, None, 256)
    bi_gru = Dropout(0.5, name='bigru_dropout')(bi_gru)

    mainc = TimeDistributed(Dense(nclasses), name='main_classifier')(bi_gru)
    mainc = Activation('softmax')(mainc) # (None, 36, 5)

    final_output = merge([auxc, mainc], mode='sum')

    output_model = Model(inputs=[main_input, char_input], outputs=final_output, name='output')
    return output_model


def preprocess_input_doc(docs, tokenizer, word2idx, char2idx, max_len, max_len_char_word):

    # word-level preprocessing
    tokenized_docs = []
    doc_word_idxs = []
    sent_idx = 0
    for doc_idx, doc in enumerate(docs):
        tokens = tokenizer.tokenize(doc)
        lowered_tokens = [t.lower() for t in tokens]
        tokenized_docs.append(lowered_tokens)
        token_idxs = [word2idx[word] if word in word2idx else 1 for word in lowered_tokens]     # 'UNK' idx is 1
        doc_word_idxs.append(token_idxs)
        sent_idx += 1
    padded_doc_word_idxs = pad_sequences(doc_word_idxs, maxlen=max_len)

    # char-level preprocessing from original codes, leave untouched
    char_per_word = []
    char_word = []
    char_senc = []

    for sentence in tokenized_docs:
        for word in sentence:
            for c in word.lower():
                char_per_word.append(c)
            if len(char_per_word) > 15:
                char_per_word = char_per_word[:15]
            char_word.append(char_per_word)
            char_per_word = []
        char_senc.append(char_word)
        char_word = []

    char_word_lex = []
    char_lex = []
    char_word = []
    for senc in char_senc:
        for word in senc:
            for charac in word:
                char_word_lex.append([char2idx[charac] if charac in char2idx else 1])

            char_word.append(char_word_lex)
            char_word_lex = []

        char_lex.append(char_word)
        char_word = []

    char_per_word = []
    char_per_senc = []
    char_senc = []
    for s in char_lex:
        for w in s:
            for c in w:
                for e in c:
                    char_per_word.append(e)
            char_per_senc.append(char_per_word)
            char_per_word = []
        char_senc.append(char_per_senc)
        char_per_senc = []

    pad_char_all = []
    for senc in char_senc:
        while len(senc) < max_len:
            senc.insert(0, [])
        pad_char_all.append(pad_sequences(senc, maxlen=max_len_char_word))

    pad_char_all = np.array(pad_char_all)

    return padded_doc_word_idxs, pad_char_all, tokenized_docs


def pos_tag_and_lemmatize_spacy(raw_side_effects):
    doc = nlp_spacy(raw_side_effects)
    pos_lemma = []
    for i, token in enumerate(doc):
        standardized_token = lemmatize(token.text)
        pos_lemma.append((standardized_token, token.pos_))
    return pos_lemma


def pos_tag_and_lemmatize(raw_side_effects):
    return pos_tag_and_lemmatize_spacy(raw_side_effects)


def lemmatize_single_ses(single_ses):
    lemmatized_ses = []
    for se in single_ses:
        pos_lemma = pos_tag_and_lemmatize(se.lower())
        lemmatized_ses.append(" ".join([w[0] for w in pos_lemma]))
    return lemmatized_ses


def parse_raw_side_effects_simple(raw_side_effects, single_se_vocab, window_size=5, max_ngram=3):
    if isinstance(raw_side_effects, list):
        filtered_docs = raw_side_effects
    else:
        raw_side_effects = raw_side_effects.translate(str.maketrans('', '', string.punctuation)).lower()
        pos_lemma_doc = pos_tag_and_lemmatize(raw_side_effects)
        filtered_docs = [w[0] for w in pos_lemma_doc if w[0] not in ADPS]
    all_window_indices, parsed_side_effects = [], set()
    for i in range(1, min(window_size, max_ngram) + 1):
        all_window_indices.extend(list(itertools.combinations(range(window_size), i)))
    for i in range(0, len(filtered_docs)-window_size+1):
        parsing_window = filtered_docs[i: i+window_size]
        for window_indices in all_window_indices:
            tokens = [parsing_window[j] for j in window_indices]
            candidate = " ".join(tokens)
            if candidate in single_se_vocab:
                parsed_side_effects.add(candidate)
    return parsed_side_effects


def build_side_effect_vocab():
    if os.path.exists(OUTPUT_SE_VOCAB_PATH):
        print("Load se vocab from {}".format(OUTPUT_SE_VOCAB_PATH))
        return load_text_as_list(OUTPUT_SE_VOCAB_PATH)
    input_single_ses = load_text_as_list(OUTPUT_SINGLE_SE_VOCAB_PATH)
    input_lemmatized_ses = set(lemmatize_single_ses(input_single_ses))
    expert_se_data = read_csv(EXPERT_SE_PATH, True, delimiter="\t")
    se_corpus = []
    for row in tqdm(expert_se_data, desc="Building se vocab"):
        raw_se = row[2][1:-1]   # remove brackets
        parsed_se = parse_raw_side_effects_simple(raw_se, input_lemmatized_ses)
        se_corpus.extend(parsed_se)
    se_count = Counter(se_corpus).most_common()
    se_vocab = set([w[0] for w in se_count])
    for se in se_vocab:
        tokens = se.split(" ")
        if len(tokens) > 0:
            se_vocab = se_vocab.union(set([" ".join(perm) for perm in itertools.permutations(tokens)]))
    save_list_as_text(se_vocab, OUTPUT_SE_VOCAB_PATH)
    return se_vocab


with open(META_PATH, "r") as f:
    meta = json.load(f)
input_word2idx = meta["word2idx"]
input_max_len = meta["max_len"]
input_max_char_len = meta["max_char_len"]
input_char2idx = meta["char2idx"]
input_vocab_size = meta["vocsize"]
input_char_vocab_size = meta["charsize"]
input_embed_dim = meta["embed_dim"]
input_char_embed_dim = meta["char_embed_dim"]
input_num_classes = meta["num_classes"]
input_idx2label = meta["idx2label"]
input_idx2label = {int(k): v for k, v in input_idx2label.items()}
input_idx2word = dict((k, v) for v, k in input_word2idx.items())
input_idx2word[0] = 'PAD'
input_idx2word[1] = 'UNK'

print('Loading word embeddings...')
_ = glove2word2vec(GLOVE_PATH, GLOVE_TMP_PATH)
w2v = KeyedVectors.load_word2vec_format(GLOVE_TMP_PATH, binary=False, unicode_errors='ignore')
print('word embeddings loading done!')

embeds, num_unk, unk_words = init_embedding_weights(input_idx2word, w2v)
model = build_model(input_max_len, input_max_char_len, input_idx2word, w2v,
                    input_vocab_size, input_char_vocab_size, input_embed_dim, input_char_embed_dim,
                    input_num_classes, embeds)
model_checkpoint = ADR_MODEL_PATH
model.load_weights(model_checkpoint)


def parse_se(input_docs, se_vocab):
    input_tokenizer = TweetTokenizer()
    adr_label, adr_window = "I-ADR", 4

    input_word_idxs, input_char_idxs, tokenized_docs = preprocess_input_doc(input_docs, input_tokenizer,
                                                                            input_word2idx, input_char2idx,
                                                                            input_max_len,
                                                                            input_max_char_len)

    pred_probs = model.predict([input_word_idxs, input_char_idxs], verbose=0)
    pred = np.argmax(pred_probs, axis=2)

    all_extracted_ses = []
    for doc_idx, doc_tokens in enumerate(tokenized_docs):
        doc_extracted_ses = set()
        sentlen = len(doc_tokens)
        startind = input_max_len - sentlen
        pred_tokens = [input_idx2label[j] for j in pred[doc_idx][startind:]]
        for token_idx, token in enumerate(doc_tokens):
            pred_token = pred_tokens[token_idx]
            if pred_token == adr_label:
                lemmatized_doc_tokens = [lemmatize(w) for w in doc_tokens]
                cleaned_doc_tokens = [w for w in lemmatized_doc_tokens if w not in set(EXCLUDED_WORDS).union(ADPS)]
                min_window = max(0, token_idx-adr_window)
                max_window = min(token_idx+adr_window, len(cleaned_doc_tokens))
                parsing_window = cleaned_doc_tokens[min_window: max_window]
                parsed_ses = parse_raw_side_effects_simple(parsing_window, se_vocab, window_size=len(parsing_window))
                doc_extracted_ses = doc_extracted_ses.union(parsed_ses)
        all_extracted_ses.append(doc_extracted_ses)
    return all_extracted_ses


def parse_pod_se():
    if os.listdir(OUTPUT_SE_DOC_DIR):
        print("Parsed POD exist in {}".format(OUTPUT_SE_DOC_DIR))
        return True

    batch_idxs = list(range(20, 40))
    buffer_size = 10
    se_vocab = build_side_effect_vocab()

    parsed_se_data = []
    skip_nums = 0
    for batch_idx in tqdm(batch_idxs, desc="Parsing side effects from review sentences"):
        curr_idx = 0
        review_sent_path = os.path.join(OUTPUT_REVIEW_SENT_DIR, "review_{}.tsv".format(batch_idx))
        review_sents = read_csv(review_sent_path, True, "\t")
        while curr_idx < len(review_sents):
            next_curr_idx = min(len(review_sents), curr_idx + buffer_size)
            buffer_data = review_sents[curr_idx: next_curr_idx]
            buffer_text = [x[2] for x in buffer_data]
            try:
                all_extracted_ses = parse_se(buffer_text, se_vocab)
                for buffer_idx, buffer_content in enumerate(buffer_data):
                    parsed_se_data.append([curr_idx + buffer_idx, buffer_content[0], buffer_content[1],
                                           list(all_extracted_ses[buffer_idx])])
            except Exception as e:
                print(e)
                skip_nums += 1
                print("Skipped {}".format(skip_nums))
            curr_idx += buffer_size
            if curr_idx % 10 == 0:
                print("Parsed {}".format(curr_idx))
        output_batch_path = os.path.join(OUTPUT_SE_DOC_DIR, "parsed_review_{}.tsv".format(batch_idx))
        print("Save parsed batch review to {}".format(output_batch_path))
        write_csv(parsed_se_data, None, output_batch_path, delimiter="\t")


def clean_umls(raw_umls_annotations, se_vocab):
    sentences = raw_umls_annotations.split("#")
    all_parsed_ses = set()
    for sent in sentences:
        valid_tokens = [lemmatize(tok.split(":")[0]) for tok in sent.split(" ") if (":" in tok
                                                                                    and not tok.startswith("neg"))]
        parsed_ses = parse_raw_side_effects_simple(valid_tokens, se_vocab, window_size=len(valid_tokens))
        all_parsed_ses = all_parsed_ses.union(parsed_ses)
    return all_parsed_ses


def clean_umls_annotation():
    if os.listdir(OUTPUT_UMLS_DOC_DIR):
        print("Umls annotation exist in {}".format(OUTPUT_UMLS_DOC_DIR))
        return True
    se_vocab = build_side_effect_vocab()
    author_symptoms = read_csv(AUTHOR_DOC_SYMPTOMS_PATH, True, "\t")
    clean_umls_data = {}
    for row in tqdm(author_symptoms, desc="Cleaning UMLS annotations"):
        author_id, doc_id, doc_text = row[0], row[1], row[2]
        parsed_ses = clean_umls(doc_text, se_vocab)
        if len(parsed_ses) > 0:
            if doc_id not in clean_umls_data:
                clean_umls_data[doc_id] = {}
            if author_id not in clean_umls_data[doc_id]:
                clean_umls_data[doc_id][author_id] = set()
            clean_umls_data[doc_id][author_id] = clean_umls_data[doc_id][author_id].union(parsed_ses)
    for doc_id, doc_data in clean_umls_data.items():
        if len(doc_data) > 0:
            for author_id, ses in doc_data.items():
                doc_data[author_id] = list(ses)
            with open(os.path.join(OUTPUT_UMLS_DOC_DIR, "{}.json".format(doc_id)), "w") as f:
                json.dump(doc_data, f)
    return True


def split_pod_into_sents():
    if len(os.listdir(OUTPUT_REVIEW_SENT_DIR)) > 0:
        print("Review sentences exists in {}".format(OUTPUT_REVIEW_SENT_DIR))
        return True
    author_review = read_csv(AUTHOR_DOC_REVIEW_PATH, True, delimiter="\t", quotechar=None)
    batch_size = 10000
    batch_num = 0
    batched_data = []
    tokenizer = TweetTokenizer()
    doc_data_map = {}
    for row in tqdm(author_review, desc="Spliting POD reviews into sentences"):
        author_id, doc_id, doc_text = row[0], row[1], row[2]
        if doc_id not in doc_data_map:
            umls_doc_path = os.path.join(OUTPUT_UMLS_DOC_DIR, "{}.json".format(doc_id))
            if not os.path.exists(umls_doc_path):
                continue
            with open(umls_doc_path, "r") as f:
                doc_data_map[doc_id] = json.load(f)
        if author_id not in doc_data_map[doc_id]:
            continue
        valid_ses = doc_data_map[doc_id][author_id]
        valid_ses_vocab = set()
        for se in valid_ses:
            valid_ses_vocab = valid_ses_vocab.union(set(se.split(" ")))
        sents = sent_tokenize(doc_text)
        for sent in sents:
            valid_word_tokens = [t for t in tokenizer.tokenize(sent) if t in valid_ses_vocab]
            if len(valid_word_tokens) > 0:
                batched_data.append([author_id, doc_id, sent])
                if len(batched_data) == batch_size:
                    output_batch_path = os.path.join(OUTPUT_REVIEW_SENT_DIR, "review_{}.tsv".format(batch_num))
                    print("Save batch review to {}".format(output_batch_path))
                    write_csv(batched_data, None, output_batch_path, delimiter="\t")
                    batch_num += 1
                    batched_data = []
    output_batch_path = os.path.join(OUTPUT_REVIEW_SENT_DIR, "review_{}.tsv".format(batch_num))
    print("Save batch review to {}".format(output_batch_path))
    write_csv(batched_data, None, output_batch_path, delimiter="\t")


def extract_valid_docs():
    if os.path.exists(OUTPUT_VALID_DOC_PATH):
        print("Valid docs exists in {}".format(OUTPUT_VALID_DOC_PATH))
        return load_text_as_list(OUTPUT_VALID_DOC_PATH)
    clean_umls_annotation()
    valid_docs = [d.replace(".json", "") for d in tqdm(os.listdir(OUTPUT_UMLS_DOC_DIR), desc="Extracting valid docs")]
    save_list_as_text(valid_docs, OUTPUT_VALID_DOC_PATH)
    return valid_docs


def build_drug_se_map():
    if os.path.exists(OUTPUT_DRUG_SE_MAP_PATH):
        print("Load drug se map from {}".format(OUTPUT_DRUG_SE_MAP_PATH))
        return read_json(OUTPUT_DRUG_SE_MAP_PATH)
    se_vocab = build_side_effect_vocab()
    expert_se_data = read_csv(EXPERT_SE_PATH, True, delimiter="\t")
    drug_se_map = {}
    tokenizer = TweetTokenizer()
    for row in tqdm(expert_se_data, desc="Building drug se map"):
        raw_se = row[2][1:-1]  # remove brackets
        se_type = row[1].lower()
        drug_names = [n for n in tokenizer.tokenize(row[0]) if n not in EXCLUDED_WORDS]
        parsed_se = parse_raw_side_effects_simple(raw_se, se_vocab)
        for name in drug_names:
            if name not in drug_se_map:
                drug_se_map[name] = {
                    "more common": set(),
                    "less common": set(),
                    "rare": set(),
                    "not known": set(),
                    "overdose": set()
                }
            drug_se_map[name][se_type] = drug_se_map[name][se_type].union(parsed_se)
    for drug_name, se_data in drug_se_map.items():
        for se_type, ses in se_data.items():
            drug_se_map[drug_name][se_type] = list(ses)
    write_json(drug_se_map, OUTPUT_DRUG_SE_MAP_PATH)
    return drug_se_map


def build_standard_se_map():
    if os.path.exists(OUTPUT_STANDARD_SE_MAP_PATH):
        print("Standard se map exists in {}".format(OUTPUT_STANDARD_SE_MAP_PATH))
        return read_json(OUTPUT_STANDARD_SE_MAP_PATH)
    se_vocab = build_side_effect_vocab()
    standard_se_map = {}
    for se in tqdm(se_vocab, desc="Standardize side effects"):
        se_tokens = se.split(" ")
        all_perms = list(itertools.permutations(se_tokens))
        matched = False
        for perm in all_perms:
            perm_str = " ".join(perm)
            if perm_str in standard_se_map:
                standard_se_map[se] = perm_str
                matched = True
                break
        if not matched:
            standard_se_map[se] = se
    write_json(standard_se_map, OUTPUT_STANDARD_SE_MAP_PATH)


if __name__ == "__main__":
    build_side_effect_vocab()
    clean_umls_annotation()
    split_pod_into_sents()
    parse_pod_se()
    extract_valid_docs()
    build_drug_se_map()
    build_standard_se_map()
