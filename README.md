## 使用
train.py用于训练模型
参数介绍：
- save-dir：字符串类型，设置训练结果和验证结果的保存地址
  - 注： 不支持空文件夹创建，所以要提前建好文件夹，自己跑的时候设置好就行了
- weights-path：字符串类型，预训练模型地址
- data：数据集路径，使用的yaml文件设置的，参考在config文件夹内的data.yan=ml
- hyp: 同上
- 其余参数根据字面意思便可知道

predict.py用于预测
- 各参数设置见上

注释写的还算清楚，按着顺序看应该没问题

## 超参数设置
- lr_min: le4
- lr_max: le2
- epoch: ？
- batch_size: ？
- optimizer：Adam

## 数据集
数据集划分：按6：2：2的比例进行的划分，采用的是提前划分好的方式，可以看dataset文件夹，将整个数据集分成了train、eval、test三部分

## 数据处理方式：
1. 图像尺寸同一缩放到了 ？？
2. 训练过程中只使用了反转，旋转这两个增强方式，每种触发概率为0.5
3. 数值未进行处理直接输入和图像传入网络中


## 结果
实验结果全部存放在了result文件夹下：

    result
      |-----image : 存放仅使用图像的实验结果
          |------ train 训练过程中各项指标
          |------ eval  验证过程中各项指标
          |------ test  模型对测试集的结果：混淆矩阵图以及各项指标文件
      |-----image_value ：同上

acc、loss图：存放在result文件夹下不同模式的train/eval中

混淆矩阵图：存放在result文件夹下不同模式的test中：confusion_matrix.jpg

精确率、准确率、召回率、F1分数：存放在result文件夹下不同模式的test中：metric.txt


注意：我用的GPU是3060 6G