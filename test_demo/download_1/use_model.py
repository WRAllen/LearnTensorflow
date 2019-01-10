import tensorflow as tf
from sklearn.preprocessing import Normalizer
import numpy as np
import matplotlib.pyplot as plt

### 从文件获取测试数据
train_data = np.loadtxt(open("data.csv", "rb"), delimiter=",", skiprows=1)

tezheng2 = train_data[:100, [0, 2]]
## 销量结果，用于训练
sale = train_data[:100,1]
X = Normalizer().fit_transform(tezheng2)
y_pred_1 = sale.reshape((-1,1))
print(y_pred_1)

sale_1 = train_data[:100,1]
y_train = sale_1.reshape((-1,1)) 
  
with tf.Session() as sess:
    # 拿到 图的元信息 saver
    saver = tf.train.import_meta_graph(meta_graph_or_file="nn_boston_model001/nn_boston.model-160000.meta")          
    # 这个也很重要，虽然还不知道是做什么的
    model_file = tf.train.latest_checkpoint(checkpoint_dir="nn_boston_model001")
    # 执行恢复
    saver.restore(sess=sess, save_path=model_file)
    # 此处得到的是图结构
    graph = tf.get_default_graph()
    # get placeholder from graph  拿到两个需要输入的数据节点，预测时输入
    inputX = graph.get_tensor_by_name("inputX:0") #输入x    
    keep_prob_s = graph.get_tensor_by_name("keep_prob:0")
    # get operation from graph   拿到能输出结果的数据节点，预测时执行到这里，拿到预测结果值
    prediction = graph.get_tensor_by_name("pred:0")
    # 开始预测, 还是不用输入y , 丢弃我一直写的1，不丢弃
    feed_dict = {inputX: X, keep_prob_s: 1}
    y_pred = sess.run(prediction, feed_dict=feed_dict)
    #### draw pics  这里画了一个坐标图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)           # 第一个块里画
    ax.plot(range(100), y_train[0:100], 'b')  # 先画出样本前50个数据的真实结果
    ax.plot(range(100), y_pred, 'r')
    ax.set_ylim([0, 6])                    # 设置纵轴的范围
    plt.ion()                               # 打开交互模式，不打开不能在训练的过程中画图
    plt.show()
#    #计算准确率
#    print(y_pred, y_pred_1)
#    correct_prediction = tf.equal(y_pred, y_pred_1)
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#    accuracy_score = sess.run(accuracy, feed_dict={inputX: X, y_true:y_pred_1})
#    print("测试数据的准确度:%f"%(accuracy_score))
#    correct_prediction = tf.equal(tf.argmax(y_pred, y_train))
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#    print('Accuracy:', accuracy.eval({inputX: X, y_true:y_train}))
    #得到预测结果
#    print(y_pred)

#pred_feed_dict = {inputX: X_train, keep_prob_s: 1}      # 用来预测的数据,不需要y
#            pred = sess.run(prediction, feed_dict=pred_feed_dict)   # 走到prediction即可 
#            ## 将预测的结果画到图像上,与真实值做对比
#            lines = ax.plot(range(50), pred[0:50], 'r--')
#            try:
#                ax.lines.remove(lines[0])
#            except:
#                pass
#            plt.pause(1)