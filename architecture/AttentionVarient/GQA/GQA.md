1. 为什么使用MQA， GQA
推理阶段需要使用kv cache，但会带来问题，https://zhuanlan.zhihu.com/p/686149289
比如L层，b * n * d的输入，缓存量就是
L*b*n*d（num of heads * head size）*2(k\v)*2(2byte)会超过gpu缓存上限。所以就借助GQA MQA。gqa是综合效果最好的。
缓存量变为：L*b*n*（num of heads * head size）*2(k\v)*2(2byte)


另一个角度，从Arithmetic intensity（越高越好）
![alt text](9634123f-b3f6-40c2-9245-a9841e96a517.png)
![alt text](23ee6a33-e589-48e6-b0f7-84306247517f.png)
![alt text](58a550ca-8ef7-4dcf-a528-e05f82374c1e.png)
![alt text](45e654db-a9d0-4a12-8625-85093039d681.png)

2. MQA vs GQA
![alt text](a424e8fe-1f76-42ed-b0df-3b1e86a0bcae.png)
![alt text](01a8cb7f-4e18-4e55-a45a-cd934da75b54.png)
