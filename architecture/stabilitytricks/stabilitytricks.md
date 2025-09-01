![alt text](ae890356-3008-4ca8-a7d6-2022defe51c7.png)
可以看到，蓝色虽然收敛了，但其实有很多突刺，不稳定，容易梯度爆炸

什么导致不稳定？
1. softmax
![alt text](c5c38cbf-aefd-43d9-ab48-51fd88859e4d.png)
可以对softmax输入做norm，比如qk结果做qk-norm
![alt text](328cd84b-8da2-465a-93d0-67a105369e63.png)