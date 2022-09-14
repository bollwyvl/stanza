"""
Given a text file and a file with one word per line, convert the text file

Sentence splits should be represented as blank lines at the end of a sentence.
"""

import argparse
import random

from stanza.models.tokenization.utils import match_tokens_with_text
import stanza.utils.datasets.common as common

def read_tokens_file(token_file):
    """
    Returns a list of list of tokens

    Each sentence is a list of tokens
    """
    sentences = []
    current_sentence = []
    with open(token_file, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(line)
        if current_sentence:
            sentences.append(current_sentence)

    return sentences

def read_sentences_file(sentence_file):
    sentences = []
    with open(sentence_file, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sentences.append(line)
    return sentences

def process_raw_file(text_file, token_file, sentence_file):
    """
    Process a text file separated into a list of tokens using match_tokens_with_text from the tokenizer

    The tokens are one per line in the token_file
    The tokens in the token_file must add up to the text_file modulo whitespace.

    Sentences are also one per line in the sentence_file
    These must also add up to text_file

    The return format is a list of list of conllu lines representing the sentences.
    The only fields set will be the token index, the token text, and possibly SpaceAfter=No
    where SpaceAfter=No is true if the next token started with no whitespace in the text file
    """
    with open(text_file, encoding="utf-8") as fin:
        text = fin.read()

    tokens = read_tokens_file(token_file)
    tokens = [[token for sentence in tokens for token in sentence]]
    tokens_doc = match_tokens_with_text(tokens, text)

    assert len(tokens_doc.sentences) == 1
    assert len(tokens_doc.sentences[0].tokens) == len(tokens[0])

    sentences = read_sentences_file(sentence_file)
    sentences_doc = match_tokens_with_text([sentences], text)

    assert len(sentences_doc.sentences) == 1
    assert len(sentences_doc.sentences[0].tokens) == len(sentences)

    start_token_idx = 0
    sentences = []
    for sent_idx, sentence in enumerate(sentences_doc.sentences[0].tokens):
        tokens = []
        tokens.append("# sent_id = %d" % (sent_idx+1))
        tokens.append("# text = %s" % text[sentence.start_char:sentence.end_char].replace("\n", " "))
        token_idx = 0
        while token_idx + start_token_idx < len(tokens_doc.sentences[0].tokens):
            token = tokens_doc.sentences[0].tokens[token_idx + start_token_idx]
            if token.start_char >= sentence.end_char:
                # have reached the end of this sentence
                # continue with the next sentence
                start_token_idx += token_idx
                break

            if token_idx + start_token_idx == len(tokens_doc.sentences[0].tokens) - 1:
                # definitely the end of the document
                space_after = True
            elif token.end_char == tokens_doc.sentences[0].tokens[token_idx + start_token_idx + 1].start_char:
                space_after = False
            else:
                space_after = True
            token = [str(token_idx+1), token.text] + ["_"] * 7 + ["_" if space_after else "SpaceAfter=No"]
            assert len(token) == 10, "Token length: %d" % len(token)
            token = "\t".join(token)
            tokens.append(token)
            token_idx += 1
        sentences.append(tokens)
    return sentences

def extract_sentences(text_files, token_files, sentence_files):
    sentences = []
    for text_file, token_file, sentence_file in zip(text_files, token_files, sentence_files):
        sentences.extend(process_raw_file(text_file, token_file, sentence_file))
    return sentences

def split_sentences(sentences, train_split=0.8, dev_split=0.1):
    """
    Splits randomly without shuffling
    """
    generator = random.Random(1234)

    train = []
    dev = []
    test = []
    for sentence in sentences:
        r = generator.random()
        if r < train_split:
            train.append(sentence)
        elif r < train_split + dev_split:
            dev.append(sentence)
        else:
            test.append(sentence)
    return (train, dev, test)

SHARDS = ("train", "dev", "test")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_files', type=str, default="extern_data/sindhi/tokenization/FolkStory1.txt,extern_data/sindhi/tokenization/String1.txt", help="Where to find the text files")
    parser.add_argument('--token_files', type=str, default="extern_data/sindhi/tokenization/tkns_FolkStory1.txt,extern_data/sindhi/tokenization/tkns_String1.txt", help="Where to find the token files")
    parser.add_argument('--sentence_files', type=str, default="extern_data/sindhi/tokenization/stns_FolkStory1.txt,extern_data/sindhi/tokenization/stns_String1.txt", help="Where to find the sentence files")
    parser.add_argument('--output_path', type=str, default="data/tokenize", help="Where to output the results")
    parser.add_argument('--dataset', type=str, default="sd_isra", help="What name to give this dataset")
    args = parser.parse_args()

    text_files = args.text_files.split(",")
    token_files = args.token_files.split(",")
    sentence_files = args.sentence_files.split(",")

    tokenizer_dir = args.output_path
    short_name = args.dataset  # todo: convert a full name?

    if len(text_files) != len(token_files):
        raise ValueError("Expected same number of text and token files")
    if len(text_files) != len(sentence_files):
        raise ValueError("Expected same number of text and sentence files")

    sentences = extract_sentences(text_files, token_files, sentence_files)
    splits = split_sentences(sentences)

    for dataset, shard in zip(splits, SHARDS):
        output_conllu = common.tokenizer_conllu_name(tokenizer_dir, short_name, shard)
        common.write_sentences_to_conllu(output_conllu, dataset)

    common.convert_conllu_to_txt(tokenizer_dir, short_name)
    common.prepare_tokenizer_treebank_labels(tokenizer_dir, short_name)

if __name__ == '__main__':
    main()
