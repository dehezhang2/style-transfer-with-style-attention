import os
import random
path1 = './style_set/val2014/'
path2 = './content_set/val2014/'
path3 = './content_style_mix_set/val2014/'
list1 = random.sample(os.listdir(path1), 20000)
list2 = random.sample(os.listdir(path2), 20000)
num = 0
for img in list1:
    if num%1000 == 0:
        print(str(num)+' finished')
    num += 1
    os.system('cp '+path1+img+' '+path3)
for img in list2:
    if num%1000 == 0:
        print(str(num)+' finished')
    num += 1
    os.system('cp '+path2+img+' '+path3)