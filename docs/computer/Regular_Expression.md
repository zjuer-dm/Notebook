# 正则表达式
<p>正则表达式(Regular Expression)是一种文本模式，包括普通字符（例如，a 到 z 之间的字母）和特殊字符（称为"元字符"），可以用来描述和匹配字符串的特定模式。</p>

1. 正则表达式是一种用于模式匹配和搜索文本的工具。  
2. 正则表达式提供了一种灵活且强大的方式来查找、替换、验证和提取文本数据。
3. 正则表达式可以应用于各种编程语言和文本处理工具中，如 JavaScript、Python、Java、Perl

## 正则表达式的简介
正则表达式的模式可以包括以下内容：

1. 字面值字符：例如字母、数字、空格等，可以直接匹配它们自身。

2. 特殊字符：例如点号 .、星号 *、加号 +、问号 ? 等，它们具有特殊的含义和功能。

3. 字符类：用方括号 [ ] 包围的字符集合，用于匹配方括号内的任意一个字符。

4. 元字符：例如 \d、\w、\s 等，用于匹配特定类型的字符，如数字、字母、空白字符等。

5. 量词：例如 {n}、{n,}、{n,m} 等，用于指定匹配的次数或范围。

6. 边界符号：例如 ^、$、\b、\B 等，用于匹配字符串的开头、结尾或单词边界位置。

## 字符匹配

### 普通字符
普通字符按照字面意义进行匹配，例如匹配字母 "a" 将匹配到文本中的 "a" 字符。

### 元字符
元字符具有特殊的含义，例如：

- `\d` 匹配任意数字字符,等价于 [0-9]
- `\w` 匹配任意字母数字字符 , 匹配字母、数字、下划线。等价于 [A-Za-z0-9_]
- `.` 匹配任意字符（除了换行符）
- `[\s\S]` 匹配所有。\s 是匹配所有空白符，包括换行，\S 非空白符，不包括换行。

### 量词
- `*`：匹配前面的模式零次或多次。
- `+`：匹配前面的模式一次或多次。
- `?`：匹配前面的模式零次或一次。
- `{n}`：匹配前面的模式恰好 n 次。
- `{n,}`：匹配前面的模式至少 n 次。
- `{n,m}`：匹配前面的模式至少 n 次且不超过 m 次。

### 字符类
- `[ ]`：匹配括号内的任意一个字符。例如，`[abc]` 匹配字符 "a"、"b" 或 "c"。
- `[^ ]`：匹配除了括号内的字符以外的任意一个字符。例如，`[^abc]` 匹配除了字符 "a"、"b" 或 "c" 以外的任意字符。

### 边界匹配
- `^`：匹配字符串的开头。
- `$`：匹配字符串的结尾。
- `\b`：匹配单词边界。
- `\B`：匹配非单词边界。

\b	匹配一个单词边界，也就是指单词和空格间的位置。例如， 'er\b' 可以匹配"never" 中的 'er'，但不能匹配 "verb" 中的 'er'。

\B	匹配非单词边界。'er\B' 能匹配 "verb" 中的 'er'，但不能匹配 "never" 中的 'er'。

### 分组和捕获
- `( )`：用于分组和捕获子表达式。
- `(?: )`：用于分组但不捕获子表达式。

### 特殊字符
- `\`：转义字符，用于匹配特殊字符本身。
- `.`：匹配任意字符（除了换行符）。
- `|`：用于指定多个模式的选择。


### 先行断言(lookahead)和后行断言(lookbehind)
- `(?=pattern)` 零宽正向先行断言(zero-width positive lookahead assertion)
代表字符串中的一个位置，紧接该位置之后的字符序列能够匹配 pattern。

例如对 "a regular expression" 这个字符串，要想匹配 regular 中的 re，但不能匹配 expression 中的 re，可以用 re(?=gular)，该表达式限定了 re 右边的位置，这个位置之后是 gular，但并不消耗 gular 这些字符。

- `(?!pattern)` 零宽负向先行断言(zero-width negative lookahead assertion)

代表字符串中的一个位置，紧接该位置之后的字符序列不能匹配 pattern。

例如对 "regex represents regular expression" 这个字符串，要想匹配除 regex 和 regular 之外的 re，可以用 re(?!g)，该表达式限定了 re 右边的位置，这个位置后面不是字符 g。
>负向和正向的区别，就在于该位置之后的字符能否匹配括号中的表达式。

- `(?<=pattern)` 零宽正向后行断言(zero-width positive lookbehind assertion)

代表字符串中的一个位置，紧接该位置之前的字符序列能够匹配 pattern。

例如对 regex represents regular expression 这个字符串，有 4 个单词，要想匹配单词内部的 re，但不匹配单词开头的 re，可以用 (?<=\w)re，单词内部的 re，在 re 前面应该是一个单词字符。
>之所以叫后行断言，是因为正则表达式引擎在匹配字符串和表达式时，是从前向后逐个扫描字符串中的字符，并判断是否与表达式符合，当在表达式中遇到该断言时，正则表达式引擎需要往字符串前端检测已扫描过的字符，相对于扫描方向是向后的。

- `(?<!pattern)` 零宽负向后行断言(zero-width negative lookbehind assertion)

代表字符串中的一个位置，紧接该位置之前的字符序列不能匹配 pattern。

例如对 "regex represents regular expression" 这个字符串，要想匹配单词开头的 re，可以用 (?<!\w)re。单词开头的 re，在本例中，也就是指不在单词内部的 re，即 re 前面不是单词字符。当然也可以用 \bre 来匹配。

## 正则表达式 - 修饰符
- i ignore - 不区分大小写
- g global - 全局匹配
- m	multi line - 多行匹配
- s	特殊字符圆点 . 中包含换行符 \n

### exmple
```javascript
//1
var str="Google runoob taobao runoob"; 
var n1=str.match(/runoob/);   // 查找第一次匹配项
var n2=str.match(/runoob/g);  // 查找所有匹配项

//2
var str="Google runoob taobao RUNoob"; 
var n1=str.match(/runoob/g);   // 区分大小写
var n2=str.match(/runoob/gi);  // 不区分大小写
//3
<head>
<meta charset="utf-8">
<title>Try</title>
</head>
<body>
<p>单击按钮显示匹配项：</p>
<p id="demo1"></p>
<p id="demo2"></p>
<button onclick="myFunction()">点我</button>
<script>
function myFunction(){
	var str="runoobgoogle\ntaobao\nrunoobweibo";
    var n1=str.match(/^runoob/g);   // 匹配一个
    var n2=str.match(/^runoob/gm);  // 多行匹配
	document.getElementById("demo1").innerHTML=n1;
	document.getElementById("demo2").innerHTML=n2;
}
</script>

</body>
</html>

//4
var str="google\nrunoob\ntaobao";
var n1=str.match(/google./);   // 没有使用 s，无法匹配\n
var n2=str.match(/runoob./s);  // 使用 s，匹配\n
```
## 运算符优先级
1. \	转义符
2. (), (?:), (?=), []	圆括号和方括号
3. *, +, ?, {n}, {n,}, {n,m}	限定符
4. ^, $, \任何元字符、任何字符	定位点和序列（即：位置和顺序）
5. |	替换，"或"操作字符具有高于替换运算符的优先级，使得"m|food"匹配"m"或"food"。若要匹配"mood"或"food"，请使用括号创建子表达式，从而产生"(m|f)ood"。