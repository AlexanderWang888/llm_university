如何让context更长？（时间复杂度和算力）
1. sparse window attention
![alt text](26b67368-3aee-4970-89f6-1f0ddaa0e316.png)
2. sliding window attention
![alt text](2c272b1d-4f57-48df-be8c-91c54f44ab50.png)
3. Current standard trick （因为普通的selfattention模块少了，所以优化了性能；同时还加入了sliding window）
（SWA结合RoPE（旋转位置嵌入）有效处理短距离依赖，提供高效的局部上下文理解。
每第四个全注意力块（Full）捕捉全局依赖，确保模型不会丢失长距离信息（如文档开头和结尾之间的关系），提高整体语义理解能力。)
(no-positional embedding，一方面是因为rope本来就是在注意力层进行的，不是在embedding时进行的)
![alt text](cdfcd073-6830-4ffd-8978-d51d3f754ee5.png)