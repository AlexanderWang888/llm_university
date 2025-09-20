1. compression ratio(压缩比)
原文的字节数（字符数）/ encoding后的token数

2. tokenizer分类
a. Character-based tokenization
一个unicode字符对应一个码点，就是一个token
如果压缩比（字节计算），则压缩比还可以，但是：
问题 1：词汇量过大。
问题 2：许多字符十分罕见（例如：🌍），这造成了词汇表的低效利用。

b. Byte-based tokenization
基于utf-8把text转为一个byte 的list，然后一个bytex转为一个数字，最后词表只有256个值。
好处：词表小
坏处：The compression ratio is terrible, which means the sequences will be too long.（尤其是attention，时间复杂度是n^2)

c. Word-based tokenization
problems: 
(1)The number of words is huge (like for Unicode characters).  
(2)Many words are rare and the model won't learn much about them.
(3)This doesn't obviously provide a fixed vocabulary size.
(4)New words we haven't seen during training get a special UNK token, which is ugly and can mess up perplexity calculations.
压缩比肯定高

d. Byte Pair Encoding
start with each byte as a token, and successively merge the most common pair of adjacent tokens.
压缩比还不错，词表也不会过大