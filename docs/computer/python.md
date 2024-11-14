# Python

## 做题中的小点

1. 直接赋值：其实就是对象的引用（别名）。
2. 浅拷贝(copy)：拷贝父对象，不会拷贝对象的内部的子对象。
3. 深拷贝(deepcopy)： copy 模块的 deepcopy 方法，完全拷贝了父对象及其子对象。

```py
import copy
dict1 = {'user': 'A',
         'num': [1, 2, 3]}

dict2 = dict1         # 浅拷贝: 引用对象
dict3 = dict1.copy()  # 浅拷贝：深拷贝父对象（一级目录），子对象（二级目录）不拷贝，还是引用
dict4 = copy.deepcopy(dict1)

dict1['user'] = 'BBB'
dict1['num'].remove(1)

# 输出结果
print("dict1: ", dict1)
print("dict2: ", dict2)
print("dict3: ", dict3)
print("dict4: ", dict4)

# dict1:  {'user': 'BBB', 'num': [2, 3]}
# dict2:  {'user': 'BBB', 'num': [2, 3]}
# dict3:  {'user': 'A', 'num': [2, 3]}
# dict4:  {'user': 'A', 'num': [1, 2, 3]}

```

python中[0 ]* n与[0 for _ in range(n)]的区别与联系：

[ 0 ] * n 是浅拷贝， 也就是把一个列表重复了 n 次，是 = 号复制（注意不是浅拷贝，= 与浅拷贝的list id是不同的）；[[0]*n]*m 这种方式是直接将 [0]*n 复制了m遍

[0 for _ in range(n)] 才是创建，深拷贝

```py
m,n = 3,4
dp1 = [[0] * n ] * m
dp2 = [[0 for _ in range(n) ] for _ in range(m)]
dp3 = [[0] * n for _ in range(m)]
dp1[0][2] = 3
dp2[0][2] = 3
dp3[0][2] = 3
print('dp1:',dp1)
print('dp2:',dp2)
print('dp2:',dp3)

# dp1: [[0, 0, 3, 0], [0, 0, 3, 0], [0, 0, 3, 0]]
# dp2: [[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
# dp2: [[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

```

**sorted**

    students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
    >>> sorted(students, key=lambda s: s[2])            # 按年龄排序
    [('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
    
    >>> sorted(students, key=lambda s: s[2], reverse=True)       # 按降序
    [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]

**匿名函数：**

```py
sum = lambda arg1, arg2: arg1 + arg2

# 调用sum函数
print "相加后的值为 : ", sum( 10, 20 )
print "相加后的值为 : ", sum( 20, 20 )
```

**使用Python的Counter进行计数**

在Python中，collections模块的Counter类是一个用于计数可哈希对象的字典子类。它可以帮助你以一种高效和Python风格的方式来计数，而不需要使用传统的涉及循环和嵌套数据结构的技术。

Counter的基本用法

Counter的基本用法是将一个序列或可迭代对象作为参数传递给它的构造函数。Counter会迭代输入的序列，计算每个对象出现的次数，并将对象作为键，出现次数作为值存储在字典中。

from collections import Counter

**使用字符串初始化Counter**
counter = Counter("mississippi")
print(counter) # 输出: Counter({'i': 4, 's': 4, 'p': 2, 'm': 1})

**使用列表初始化Counter**
counter = Counter(list("mississippi"))
print(counter) # 输出: Counter({'i': 4, 's': 4, 'p': 2, 'm': 1})
更新Counter

你可以使用update()方法来更新现有的Counter。与dict的update()方法不同，Counter的update()方法会将现有计数相加，而不是替换它们。

counter = Counter({'a': 3, 'b': 2})
counter.update({'a': 1, 'c': 2})
print(counter) # 输出: Counter({'a': 4, 'b': 2, 'c': 2})
访问Counter的内容

Counter几乎具有与dict相同的接口，你可以使用类似字典的键访问（[key]）来访问它们的值。你还可以使用常见的方法和技术来迭代键、值和项。

counter = Counter("mississippi")
print(counter['s']) # 输出: 4
for letter in counter:
print(letter, counter[letter])
查找最常见的对象

如果你需要根据对象的出现频率列出一组对象，你可以使用most_common()方法。这个方法返回一个按对象当前计数排序的(对象, 计数)列表。如果计数相等，则按它们首次出现的顺序排列。

counter = Counter('mississippi')
print(counter.most_common(3)) # 输出: [('i', 4), ('s', 4), ('p', 2)]

## Python3 面向对象
* 类有一个名为 `__init__()` 的特殊方法（构造方法），该方法在类实例化时会自动调用
* 类定义了 __ init__() 方法，类的实例化操作会自动调用 __ init__() 方法
```python
class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart
x = Complex(3.0, -4.5)
print(x.r, x.i)   # 输出结果：3.0 -4.5
```

* self 代表类的实例，而非类.类的方法与普通的函数只有一个特别的区别——它们必须有一个额外的第一个参数名称, 按照惯例它的名称是 self。
```py
class Test:
    def prt(self):
        print(self)
        print(self.__class__)
 
t = Test()
t.prt()
```
- 输出：
```
<__main__.Test instance at 0x100771878>
__main__.Test
```
**从执行结果可以很明显的看出，self 代表的是类的实例，代表当前对象的地址，而 self.class 则指向类.self 不是 python 关键字，我们把他换成 runoob 也是可以正常执行的**

```py
#类定义
class people:
    #定义基本属性
    name = ''
    age = 0
    #定义私有属性,私有属性在类外部无法直接进行访问
    __weight = 0
    #定义构造方法
    def __init__(self,n,a,w):
        self.name = n
        self.age = a
        self.__weight = w
    def speak(self):
        print("%s 说: 我 %d 岁。" %(self.name,self.age))
 
#单继承示例
class student(people):
    grade = ''
    def __init__(self,n,a,w,g):
        #调用父类的构函
        people.__init__(self,n,a,w)
        self.grade = g
    #覆写父类的方法
    def speak(self):
        print("%s 说: 我 %d 岁了，我在读 %d 年级"%(self.name,self.age,self.grade))
 
 
 
s = student('ken',10,60,3)
s.speak()
```
### 多继承
- 需要注意圆括号中父类的顺序，若是父类中有相同的方法名，而在子类使用时未指定，python从左至右搜索 即方法在子类中未找到时，从左到右查找父类中是否包含方法。
```py
#!/usr/bin/python3
 
#类定义
class people:
    #定义基本属性
    name = ''
    age = 0
    #定义私有属性,私有属性在类外部无法直接进行访问
    __weight = 0
    #定义构造方法
    def __init__(self,n,a,w):
        self.name = n
        self.age = a
        self.__weight = w
    def speak(self):
        print("%s 说: 我 %d 岁。" %(self.name,self.age))
 
#单继承示例
class student(people):
    grade = ''
    def __init__(self,n,a,w,g):
        #调用父类的构函
        people.__init__(self,n,a,w)
        self.grade = g
    #覆写父类的方法
    def speak(self):
        print("%s 说: 我 %d 岁了，我在读 %d 年级"%(self.name,self.age,self.grade))
 
#另一个类，多继承之前的准备
class speaker():
    topic = ''
    name = ''
    def __init__(self,n,t):
        self.name = n
        self.topic = t
    def speak(self):
        print("我叫 %s，我是一个演说家，我演讲的主题是 %s"%(self.name,self.topic))
 
#多继承                      #Tim 说: 我 25岁了，我在读 4 年级
class sample(speaker,student):    # 如果是class sample(student,speaker):
    
    a =''
    def __init__(self,n,a,w,g,t):
        student.__init__(self,n,a,w,g)
        speaker.__init__(self,n,t)
 
test = sample("Tim",25,80,4,"Python")
test.speak()   #方法名同，默认调用的是在括号中参数位置排前父类的方法  
#  我叫 Tim，我是一个演说家，我演讲的主题是 Python
```
### 类属性与方法

类的私有属性:

__private_attrs：两个下划线开头，声明该属性为私有，不能在类的外部被使用或直接访问。在类内部的方法中使用时 self.__private_attrs。

类的方法:

在类的内部，使用 def 关键字来定义一个方法，与一般函数定义不同，类方法必须包含参数 self，且为第一个参数，self 代表的是类的实例。

self 的名字并不是规定死的，也可以使用 this，但是最好还是按照约定使用 self。

类的私有方法:

__private_method：两个下划线开头，声明该方法为私有方法，只能在类的内部调用 ，不能在类的外部调用。self.__private_methods。

## Python3 模块
在前面的几个章节中我们基本上是用 python 解释器来编程，如果你从 Python 解释器退出再进入，那么你定义的所有的方法和变量就都消失了。

为此 Python 提供了一个办法，把这些定义存放在文件中，为一些脚本或者交互式的解释器实例使用，这个文件被称为模块。

模块是一个包含所有你定义的函数和变量的文件，其后缀名是.py。模块可以被别的程序引入，以使用该模块中的函数等功能。这也是使用 python 标准库的方法。

一个模块只会被导入一次，不管你执行了多少次 import
### __name__属性
* 一个模块被另一个程序第一次引入时，其主程序将运行。如果我们想在模块被引入时，模块中的某一程序块不执行，我们可以用__name__属性来使该程序块仅在该模块自身运行时执行。
```py
#!/usr/bin/python3
# Filename: using_name.py

if __name__ == '__main__':
   print('程序自身在运行')
else:
   print('我来自另一模块')
```

## 缓存装饰器
使用缓存是优化Python程序速度的重要方法之一。如果使用得当，可以大幅减少计算资源的负载，有效加快代码运行速度

Python 的内置库 functools 模块附带了@lru_cache，@cache, @cached_property 装饰器

@lru_cache 是最常见的缓存装饰器。lru_cache 是： Last recently used cache 的简写，可以将该函数最近调用的输入参数以及结果进行缓存。如果有新的调用，先检查缓存是否有相同的输入参数，如果存在，则直接返回对应结果。如果是无参函数，第1次调用后，以后每次调用，直接返回缓存结果。

lru_cache默认不清除缓存内容，因此缓存会无限增长，如果程序是长期运行的服务，可能存在耗尽内存的风险。 因此，必须添加1个maxsize参数

```py
import functools
import gc

# 主要功能： 
# 验证  @lru_cache 装饰器，.chche_info() 和 .cache_clear() 方法的使用
#       garbage collection 的使用

@functools.lru_cache(maxsize = 300) # Max number of Last recently used cache
def fib(n):
	if n < 2:
		return n
	return fib(n-1) + fib(n-2)


fib(30)
fib.cache_clear()

# Before Clearing
print(fib.cache_info())

# After Clearing
print(fib.cache_info())

@functools.lru_cache(maxsize = None)
def gfg1():
    # insert function logic here
    pass

# 再次运行函数 
gfg1()
fib(30)
# garbage collection
gc.collect()

# All objects collected
objects = [i for i in gc.get_objects() 
           if isinstance(i, functools._lru_cache_wrapper)]

print(gfg1.cache_info())

# All objects cleared
for object in objects:
    object.cache_clear()
    
print(gfg1.cache_info())

```

@cache 装饰器更轻量化，速度更快，且是线程安全，不同线程可以调用同1个函数，缓存值可以共享。

@cached_property是一个装饰器，它将类的方法转换为属性，其值仅计算一次，然后缓存为普通属性。因此，只要实例持久存在，缓存的结果就可用，我们可以将该方法用作类的属性那样来使用