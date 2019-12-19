### chap05
- OrderedDict
```
from collections import OrderedDict
# 順番付きのdictを生成
dic = OrderedDict()
dic['a'] = 100
dic['c'] = 40
dic['b'] = 20
dic
OrderedDict([('a', 100), ('c', 40), ('b', 20)])

lis = list(dic)
lis
['a', 'c', 'b']

lis_v = list(dic.values())
lis_v
[100, 40, 20]
```
