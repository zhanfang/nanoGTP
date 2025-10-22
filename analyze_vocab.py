"""
分析 Shakespeare 文本中的字符集
"""

def analyze_vocab():
    # 读取文本
    with open('data/shakespeare/input.txt', 'r') as f:
        text = f.read()
    
    # 获取唯一字符
    chars = sorted(list(set(text)))
    
    # 打印统计信息
    print(f"Vocabulary size: {len(chars)}")
    print("\nAll characters:")
    for i, char in enumerate(chars):
        if char == '\n':
            char_display = '\\n'
        elif char == ' ':
            char_display = '[SPACE]'
        else:
            char_display = char
        print(f"{i:2d}: '{char_display}'")
    
    # 按类型分类字符
    uppercase = [c for c in chars if c.isupper()]
    lowercase = [c for c in chars if c.islower()]
    numbers = [c for c in chars if c.isdigit()]
    punctuation = [c for c in chars if not c.isalnum() and c != '\n' and c != ' ']
    special = [c for c in chars if c in ['\n', ' ']]
    
    print("\nCharacter categories:")
    print(f"Uppercase letters ({len(uppercase)}): {''.join(uppercase)}")
    print(f"Lowercase letters ({len(lowercase)}): {''.join(lowercase)}")
    print(f"Numbers ({len(numbers)}): {''.join(numbers)}")
    print(f"Punctuation ({len(punctuation)}): {''.join(punctuation)}")
    print(f"Special ({len(special)}): {' '.join(['\\n' if c == '\n' else '[SPACE]' if c == ' ' else c for c in special])}")

if __name__ == '__main__':
    analyze_vocab()