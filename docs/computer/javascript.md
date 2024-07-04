# Javascript
1. HTML 定义了网页的内容
2. CSS 描述了网页的布局
3. JavaScript 控制了网页的行为

```javascript
<!DOCTYPE html>
<html>
<head> 
<meta charset="utf-8"> 
<title>example</title> 
</head>
<body>

<h1>我的 Web 页面</h1>
<p id="myPar">我是一个段落。</p>
<div id="myDiv">我是一个div。</div>
<p>
<button type="button" onclick="myFunction()">点击这里</button>
</p>
<script>
function myFunction(){
	document.getElementById("myPar").innerHTML="你好世界！";
	document.getElementById("myDiv").innerHTML="你最近怎么样?";
}
</script>
<p>当您点击上面的按钮时，两个元素会改变。</p>

</body>
</html>
```

# HTML
```html
<!DOCTYPE html> 声明为 HTML5 文档  
<html> 元素是 HTML 页面的根元素  
<head> 元素包含了文档的元（meta）数据，如 <meta charset="utf-8"> 定义网页编码格式为 utf-8。  
<title> 元素描述了文档的标题  
<body> 元素包含了可见的页面内容  
<h1> 元素定义一个大标题  
<p> 元素定义一个段落  
```

# CSS

- CSS 指层叠样式表 (Cascading Style Sheets)
- 样式定义如何显示 HTML 元素
- 样式通常存储在样式表中
- 把样式添加到 HTML 4.0 中，是为了解决内容与表现分离的问题
- 外部样式表可以极大提高工作效率
- 外部样式表通常存储在 CSS 文件中
- 多个样式定义可层叠为一个

### id 选择器
 
ID属性不要以数字开头，数字开头的ID在 Mozilla/Firefox 浏览器中不起作用
```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"> 
<title>example</title> 
<style>
#para1
{
	text-align:center;
	color:red;
} 
</style>
</head>

<body>
<p id="para1">Hello World!</p>
<p>这个段落不受该样式的影响。</p>
</body>
</html>
```
### class 选择器
class 选择器用于描述一组元素的样式，class 选择器有别于id选择器，class可以在多个元素中使用。

class 选择器在 HTML 中以 class 属性表示, 在 CSS 中，类选择器以一个点 . 号显示：
```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"> 
<title>example</title> 
<style>
.center
{
	text-align:center;
}
</style>
</head>

<body>
<h1 class="center">标题居中</h1>
<p class="center">段落居中。</p> 
</body>
</html>
```

