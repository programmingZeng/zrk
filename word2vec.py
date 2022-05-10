from gensim.models import word2vec
import jieba
def fenci():
    '''使用jieba对文本分词，文本中的人名不分词'''
    names = ['沙瑞金', '田国富', '高育良', '侯亮平', '钟小艾', '陈岩石', '欧阳菁', '易学习', '王大路', '蔡成功', '刘庆祝',
             '孙连城', '季昌明', '丁义珍', '郑西坡', '赵东来', '高小琴', '赵瑞龙', '林华华', '陆亦可', '刘新建', ]
    for name in names:
        jieba.suggest_freq(name, True)
    with open('./data/in_the_name_of_people.txt', encoding='utf-8') as f:
        data = f.read()
        data_cut = jieba.cut(data)
        result = ' '.join(data_cut)
        result = result.encode('utf-8')
        with open('./data/in_the_name_of_segment.txt', 'wb') as f2:
            f2.write(result)
    f.close()
    f2.close()
def getmodel():
    # 加载《人民的名义》文本
    sentences = word2vec.LineSentence('./data/in_the_name_of_segment.txt')
    model = word2vec.Word2Vec(sentences, size=100, min_count=1, window=3, hs=1)
    model.save("./data/in_the_name_of_model.bin")  # 保存模型
    model.wv.save_word2vec_format("word2vec.txt", binary=False)  # 保存词向量

    '''预测人名相似度和非同类相似度'''
    print('李达康和王大路的人名相似度为', model.wv.similarity('李达康', '王大路')*100, '%')
    print('非同类相似度为', model.wv.similarity('李达康', '蛋糕')*100, '%')
    print('非同类相似度为', model.wv.similarity('王大路', '女性') * 100, '%')
    print('同类相似度为', model.wv.similarity('蛋糕', '女性') * 100, '%')
    print('同类相似度为', model.wv.similarity('你', '我') * 100, '%')
    return model
fenci()
getmodel()

