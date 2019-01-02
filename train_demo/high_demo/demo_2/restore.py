# -*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

#创建权重的随机噪点
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)
#二维卷积函数和卷积模板移动的步长（都是１表示不会遗漏任何一个点）
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#max_pool最大池化和移动的长（便是横竖两个方向以２为步长）
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')  
# 定义输入和编写输入形状将1D转回原来的2D                     
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1,28,28,1])
# 第一次卷积层提取３２中特征
W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
b_conv1 = bias_variable([32], "b_conv1")
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# 第二层卷积层提取６４种特征
W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
b_conv2 = bias_variable([64], 'b_conv2')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# 再将第二个卷积层的输出进行变形再转为1D,然后再接上一个全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024], "W_fc1")
b_fc1 = bias_variable([1024], 'b_fc1')
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 防止过拟合设置dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# 最后再接连一个softmax层
W_fc2 = weight_variable([1024, 10], 'W_fc2')
b_fc2 = bias_variable([10], 'b_fc2')
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.global_variables_initializer().run()

saver = tf.train.Saver()
saver.restore(sess, "my_net/net.ckpt")
print("weight:",sess.run(W_conv1))
# 输出
# weight: [[[[ 0.13466209 -0.02330974  0.08223451  0.12852599 -0.17451777
#      0.03383977 -0.05937636 -0.02815911 -0.00644034  0.11984196
#     -0.004246   -0.06483342  0.05029948 -0.16933686  0.19513537
#     -0.16229066  0.07783231  0.04227149 -0.00844805 -0.06943296
#     -0.09240122  0.13753992 -0.10454192 -0.01375801 -0.1340594
#      0.14314951 -0.04800652  0.06835844  0.04902757  0.16648331
#      0.00340653 -0.08197801]]

#   [[-0.103916    0.01858053 -0.01189241 -0.08670131 -0.18510325
#      0.14063054  0.07927114  0.09329136 -0.12048833 -0.05794959
#     -0.09596059 -0.03712238 -0.10247064  0.19519635  0.10349717
#      0.0729923  -0.00441519  0.0950795   0.01725687 -0.04661577
#      0.14124729  0.04814418 -0.10736185  0.10835538 -0.04789189
#     -0.05069529  0.10979497 -0.02737628  0.10694528 -0.12462971
#     -0.12444796 -0.1278925 ]]

#   [[-0.16861695  0.04419982 -0.04125305  0.04009061  0.009332
#      0.0846931   0.09125935  0.0575299  -0.03705056 -0.10441906
#      0.07225055  0.08433795 -0.04250629 -0.05288744 -0.04598719
#     -0.10903265  0.09166512 -0.09686512  0.12688637  0.00252955
#      0.0322784   0.12638596 -0.1685272  -0.01948836  0.12408811
#      0.01043154 -0.04477225 -0.10735349 -0.03229803 -0.02798937
#     -0.05134716  0.02006257]]

#   [[-0.134426    0.12859678 -0.01564787 -0.04254566  0.19153984
#     -0.16166843 -0.016871   -0.05517533  0.02525021 -0.02094
#     -0.01985506  0.1731676   0.04990938 -0.02522851  0.04949995
#     -0.00162401  0.02408561 -0.09324933 -0.03477434  0.17862168
#     -0.04841465  0.07034755 -0.01754781 -0.02376705 -0.02086763
#      0.0105141   0.03201102 -0.03462762  0.08987215 -0.06602514
#     -0.00457665 -0.0291146 ]]

#   [[-0.02596928 -0.0820798  -0.13307907 -0.0483731  -0.03665879
#      0.1679582  -0.06257785  0.05122788  0.01314534  0.05274828
#      0.11684856 -0.09103288 -0.04887714  0.00324281 -0.0079976
#      0.12653898  0.01832916  0.02909979 -0.00883954  0.07712403
#     -0.04029514 -0.09563394 -0.09218983  0.01776174  0.03500127
#     -0.02715611  0.03004353  0.06123044 -0.00381413 -0.04926645
#     -0.01542913 -0.02236424]]]


#  [[[-0.0238135   0.02727135  0.10489147  0.1302594  -0.15634085
#      0.03351276 -0.01444737 -0.1044094   0.02697372  0.0369133
#      0.00738436 -0.0959566  -0.12083974  0.00047158 -0.16869143
#      0.00435658  0.02897414 -0.05955643 -0.09222823  0.01175814
#     -0.14697847 -0.1140586   0.04547598 -0.019845    0.02995876
#      0.09872881 -0.05811908 -0.09522235  0.05475975 -0.0553324
#      0.1294725   0.05641558]]

#   [[ 0.03114823 -0.02441427  0.05920138  0.06195849  0.03522943
#     -0.07880697 -0.05568832 -0.05389342 -0.18704446  0.05150541
#      0.05320846  0.0633977  -0.01248473  0.0550943   0.07256936
#     -0.11477433 -0.05780865 -0.12022895  0.02961534 -0.03892944
#     -0.04425846 -0.17809747 -0.01588785 -0.02691377 -0.04816454
#      0.01348464  0.06989508  0.09109591  0.08926554  0.08839799
#     -0.06009059 -0.02008018]]

#   [[ 0.02142802 -0.13230671  0.06653494  0.11547297  0.08222214
#     -0.04704271 -0.06205628  0.09640929  0.00661877 -0.03428747
#      0.02078404 -0.05948159 -0.0744421   0.16026807 -0.03212119
#     -0.10448693 -0.04679533 -0.07530012 -0.08241515 -0.04587603
#      0.06366105 -0.06355    -0.06018241 -0.03472293 -0.01154743
#     -0.12401196 -0.06356196 -0.01139551 -0.10559456  0.05366809
#     -0.00467925 -0.06778569]]

#   [[ 0.1645606  -0.09116566  0.03244377  0.05455256  0.0405594
#     -0.19338772  0.01540006  0.00143311  0.13790108  0.02015721
#     -0.019772   -0.17483446  0.0644532   0.08714833  0.03614835
#      0.01239401  0.06403559 -0.18370809  0.08588151 -0.07148926
#      0.0442801  -0.19502178  0.09141452 -0.02880699 -0.19552746
#      0.04313487  0.04838512 -0.07338668 -0.02510849  0.12097668
#     -0.00299426 -0.00281633]]

#   [[ 0.10529729  0.05523375  0.0803192   0.10519319 -0.08623686
#      0.00106833 -0.01735409 -0.00855227  0.07865676 -0.01386632
#     -0.01262334  0.1894348  -0.11702022  0.04098804 -0.06588016
#     -0.0422832   0.00048643  0.01107574 -0.07768556  0.02686911
#     -0.05067796 -0.10357134 -0.00980487 -0.05845149 -0.06580144
#     -0.06700473 -0.06457805  0.01300991  0.08855291  0.19314377
#     -0.09226765  0.07419606]]]


#  [[[ 0.08247274  0.1347408   0.07237071  0.00399281  0.08600225
#      0.12676872  0.05010732  0.0636171   0.07694912 -0.13171986
#     -0.09663509 -0.09741924  0.08185724 -0.0097144  -0.06941678
#     -0.04092259 -0.12935795 -0.06769201  0.0929722  -0.04068904
#      0.10398491 -0.03831578 -0.02383496  0.11149587 -0.12651818
#      0.0428635  -0.06033356 -0.10218892  0.0025127   0.00461824
#     -0.0368746  -0.05599273]]

#   [[ 0.01284707  0.04912982 -0.04801436  0.05499163  0.05096188
#     -0.04570451 -0.01271799  0.12942743 -0.02166099  0.07528673
#     -0.1506361  -0.05048619  0.08787883 -0.02796299 -0.01909581
#     -0.06768387 -0.08175835 -0.06643695 -0.00056707 -0.12727399
#     -0.05916834  0.0090368  -0.05889846 -0.10750621  0.00494293
#      0.0609484  -0.20197633  0.18725516 -0.18253317 -0.14145967
#      0.085286    0.01933642]]

#   [[ 0.15609832  0.09132435 -0.01293222  0.01179005  0.11802504
#      0.05719267  0.1686149   0.04635616 -0.14949718  0.01762168
#      0.13998438 -0.01233819  0.0482237  -0.05847237 -0.06283792
#     -0.00561406 -0.02706325  0.00289151  0.11971702 -0.11233823
#     -0.15356243  0.02865429  0.02892194  0.15944202 -0.03275576
#      0.06275589  0.18616253  0.04617808 -0.15751565  0.00413813
#     -0.02510173 -0.01765056]]

#   [[-0.1556812  -0.04431971 -0.05226772  0.00411637 -0.01231483
#      0.16572228 -0.01444787 -0.0054452  -0.1921947   0.112137
#     -0.01554719 -0.05521042 -0.07079933 -0.14672177 -0.08819984
#      0.11635429  0.04120536 -0.06065181  0.18792044  0.12971567
#     -0.03416075 -0.05396206  0.04066295 -0.14777628 -0.00044764
#     -0.1455733   0.03815677 -0.05840168 -0.00893732 -0.10132165
#     -0.01140902  0.1706433 ]]

#   [[ 0.12729226  0.1328227  -0.03642015 -0.16562055 -0.03919646
#     -0.04584311  0.03761242  0.08327811  0.09956943 -0.08444075
#      0.0799851   0.07756137 -0.04398817  0.16479571 -0.02000857
#     -0.01997246 -0.04201505  0.06301146  0.19373693  0.10152405
#      0.05899535 -0.06954552  0.03382727 -0.07328583  0.0760899
#     -0.07846633 -0.09966299 -0.0045553   0.1634041   0.04670151
#     -0.15438369 -0.01675118]]]


#  [[[-0.01666035 -0.00932359  0.0840581   0.00900342 -0.00223353
#      0.01993656 -0.03373159 -0.09972098 -0.02059853 -0.05875696
#     -0.05609782  0.01141407  0.03404704  0.14631471  0.13072108
#      0.01211792 -0.0847152   0.06654444  0.06579142  0.10247984
#      0.03556653 -0.0733305  -0.00690371 -0.05842807 -0.16575883
#     -0.07658959 -0.01808594  0.09612428  0.1425608   0.12846662
#     -0.02867736  0.06201915]]

#   [[ 0.04005397  0.062535   -0.00141869 -0.1382122  -0.00389024
#     -0.00514456 -0.0811399  -0.01173165  0.11063681  0.07367174
#     -0.00750223 -0.01571834 -0.05431687 -0.01508025 -0.02075889
#      0.09664119 -0.12236825 -0.12515986  0.00275675  0.11121836
#     -0.09249922 -0.03248858 -0.0541324   0.00156064  0.06001945
#     -0.03991852  0.07571139  0.0135974   0.01974975 -0.01062751
#      0.01308475 -0.1225834 ]]

#   [[-0.1385512   0.11951158  0.11859241 -0.0711228  -0.07872471
#      0.07960994  0.11941127 -0.02965204 -0.09724934  0.16713858
#     -0.03432929 -0.16282615 -0.0309174  -0.08498165  0.05463934
#     -0.08689021 -0.14067596  0.11278313 -0.10964598  0.00109591
#     -0.02017967  0.0869535  -0.17464884  0.17356753 -0.09347689
#     -0.00950017  0.00814272  0.12758142  0.14296483  0.06208743
#     -0.1629379   0.02232544]]

#   [[-0.12638094 -0.04770992 -0.05078717  0.03645051  0.12098681
#     -0.01290469  0.09255274 -0.03268352  0.06841913 -0.07404751
#     -0.06542435 -0.1932658  -0.16489449 -0.06043186  0.0524218
#     -0.09879466  0.03271572 -0.01830071 -0.04155791 -0.04640232
#      0.09561782 -0.00035583  0.03749423  0.1155209  -0.06562019
#     -0.07957114  0.18004492  0.04014976 -0.05042906 -0.19260646
#     -0.06543596 -0.0696183 ]]

#   [[-0.12678656 -0.00285017 -0.12218072  0.05525405  0.05559193
#      0.06216715 -0.07161517  0.07159475  0.06093545  0.0047098
#      0.15099044 -0.0332464  -0.09579013 -0.15970978  0.04912662
#      0.1010863  -0.09493441  0.08590786 -0.12781034 -0.00639815
#      0.04887151 -0.00441488  0.04973672 -0.15809499 -0.02031425
#      0.0376427  -0.03456232 -0.12452798 -0.01360172 -0.06532887
#      0.1168307   0.14872709]]]


#  [[[-0.02603718  0.00240008  0.13128102 -0.11145949 -0.09570677
#     -0.14081359  0.10426251  0.041578   -0.02319436 -0.06316315
#      0.08897592  0.15767395 -0.03445492 -0.00455555  0.03043131
#      0.15046306 -0.09948661  0.11330406  0.16157366 -0.12050191
#      0.0835788   0.12166419 -0.05274393 -0.16095214  0.0375158
#      0.0624902  -0.08520517  0.10233987  0.08558054 -0.1539014
#      0.08797332  0.00907904]]

#   [[-0.05791     0.05131794 -0.01745227 -0.1580967   0.04979694
#     -0.15392257 -0.01928222 -0.01669193  0.09298469  0.00931904
#     -0.05272081  0.0677126  -0.00653178 -0.17552786  0.1477635
#     -0.02583439  0.02002623  0.12028753 -0.05735342 -0.05273659
#      0.11766843  0.01818888 -0.09761497  0.00710598 -0.02815727
#     -0.08963592 -0.01006865 -0.05545853  0.11053545  0.05737506
#     -0.03055632  0.14847969]]

#   [[-0.17971    -0.01423657  0.01975186  0.03172617 -0.11561575
#     -0.02670503 -0.06102744  0.05175057  0.05009621 -0.08224821
#     -0.09884657 -0.05482723 -0.01400837  0.09148107 -0.10602422
#      0.09114834  0.05819841  0.16758342  0.00110976  0.11895451
#     -0.08965766 -0.13841406 -0.13177739  0.00861917  0.15167199
#     -0.08184092  0.13350596 -0.12199039  0.0264039  -0.02466673
#     -0.0029213  -0.03868986]]

#   [[ 0.10764924 -0.14070818  0.10991707 -0.11521054  0.13916992
#      0.0579493   0.06179487 -0.07759616 -0.06351748  0.11938044
#     -0.04838855 -0.04882132  0.06472809  0.00582096 -0.05331021
#     -0.0563504   0.01909552 -0.07250507  0.02076001 -0.04676399
#      0.09376036  0.01816531 -0.01666988  0.04373975 -0.05621226
#     -0.10769486  0.10270082 -0.04842831 -0.07417441 -0.16680743
#      0.06338342 -0.04335238]]

#   [[ 0.02226626 -0.10461894 -0.02116431  0.07235784 -0.00297887
#      0.05983445 -0.06516706 -0.00972014  0.13466251 -0.13745895
#      0.14310668 -0.01206662 -0.17189725  0.04431105  0.16712074
#      0.01720797  0.0304414   0.07358288  0.10135682  0.00281439
#     -0.1127395   0.06928413  0.04048568  0.10788841  0.05576131
#      0.02603034  0.020636   -0.00022439 -0.0610485   0.00424536
#      0.03987328 -0.11782826]]]]




