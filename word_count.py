import string

def wordcount(text):
    # 去除标点符号
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator)
    
    # 转换为小写并分割成单词列表
    words = clean_text.lower().split()
    
    # 统计每个单词出现的次数
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
            
    return word_freq

# 示例
text = "Hello, world! Hello, Python. Python is great."
result = wordcount(text)
print(result)
