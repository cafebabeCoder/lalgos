# lalgos

用于日常模型研究<br>
- **plx**, **ptx**, **tfx** 用于基础模型的实现，抽象某种具体模型的写法, 或学习某个框架的写法。 如transformer. 用途： 学习某个模型结构、抽象某个模型结构，用于解task
- **utils** metric, functions, alphabet, logger等。 与tf, torch等框架无关。 可与sklearn相关. 与tf等框架有关的应放到具体框架下实现。 
- **task** 具体任务， 可能需要多种模型，可直接调动**plx**, **tfx**等模型写好的模型， 用于解决某个具体的问题。
- **rearch** 用于细节的研究， 比如attention计算时 是否归一化的可视化对比，多目标得分融合的方式可视化比较等。

```
├── data # 所有测试数据
├── __init__.py
├── kandian-alg
├── plx  # pytorch-lightning 完成的模型
├── ptx  # pytorch
├── README.md
├── research  # 对于算法细节的对比、研究、可视化
├── tasks # 某个任务， 如视频号推荐大赛， qq浏览器视频相似度大赛
├── test.py
├── tfx  # tensorflow
└── utils  # 常用工具
```
 