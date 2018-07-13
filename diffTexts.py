# coding:utf-8
from os import path
import MeCab
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

#比較用テキストファイル取得
current_dir = path.dirname(__file__)
diff1 = open(path.join(current_dir, 'diff1.txt'), 'r').read()
diff2 = open(path.join(current_dir, 'diff2.txt'), 'r').read()
texts = (diff1 + '|' + diff2).split('|')
print(texts)

def tokenizer(text):
    """
    文章から単語を抽出
    """
    out_word = []
    # 形態素解析
    tagger = MeCab.Tagger('-Ochasen')
    tagger.parse('')
    node = tagger.parseToNode(text)

    while node:
        word_type = node.feature.split(",")[0]
        if word_type in ["名詞"]:
            out_word.append(node.surface)
        node = node.next
    return out_word

# 学習データとなる各文書
training_docs = []
for i, text in enumerate(texts):
    training_docs.append(TaggedDocument(words=tokenizer(text), tags=['doc' +str(i + 1)]))

#print(training_docs)
# min_count=1:最低1回出現した単語を学習に使用
# dm=0: 学習モデル=DBOW
model = Doc2Vec(documents=training_docs, min_count=1, dm=0)

# モデルのセーブ
# model.save("model/doc2vec.model")

# モデルのロード
# model = Dec2Vec.load("model/doc2vec.model")

print(len(training_docs))

print(model.docvecs.most_similar('doc1'))
print(model.docvecs.most_similar('doc2'))

#for v, k in training_docs:
#    print(k)
#    print(v)
#
#    print(model.docvecs.most_similar(k))
#    for items in model.docvecs.most_similar(k):
#        print("\t" + str(items[0]) + " : "+ str(items[1]))
#    print("[" + v + "]")
#    for items in model.docvecs.most_similar(k):
#        print("\t" + tags[items[0]] + " : "+ str(items[1]))
