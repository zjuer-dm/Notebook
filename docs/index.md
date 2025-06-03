---
comments: true
nostatistics: true
hide:
  - navigation
  - toc
---

<script async src="https://www.googletagmanager.com/gtag/js?id=G-D2WBR8B2E1"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js'， new Date());

  gtag('config'， 'G-D2WBR8B2E1');
</script>
<h1 style="font-family:arial;text-align:center;">Hi,这是繁星若尘(zjuer-dm)的笔记本^_^</h1>
<p style="font-family:arial;color:grey;font-size:20px;text-align:center;">Mens et Manus</p>
<p style="font-family:arial;color:grey;font-size:20px;text-align:center;">思想与行动</p>



<p>在大二下，偶然发现了同学们用mkdocs做的笔记，不禁让人耳目一新，从此慢慢开始使用基于 MkDocs 的笔记本。</p>
<a style="font-family:arial;color:grey;font-size:20px;text-align:center;" href="https://isshikihugh.github.io/zju-cs-asio/">ZJU CS - All Sum in One!</a>
<p>在暑假，我开始在Gitpages上部署，这个笔记从2024.07.02 09:10:00开始使用,来记录一些有用的东西。</p>
<p> </p>
<p>2024年11月，老师们教导我们，有一些令人印象深刻的话：</p>
<p>真正的平庸不是你还没有做成什么，而是你放弃了追求卓越的心。</p>
<p> </p>
<p>什么都无法舍弃，就什么都无法改变。       --2025.4.2</p>


<a href="javascript:toggle_statistics();" style="font-size: 18px;color:grey;  font-family: 'Arial', sans-serif; border-bottom: 1px solid #0066cc; text-decoration: none;">站点统计</a>


<div id="statistics" markdown="1" class="card" style="width: 27em; border-color: transparent; opacity: 1; font-size: 75%">
<div style="padding-left: 1em; font-size: 14px; " markdown="1">
网站运行时间：<span id="web-time"></span>
</div>
</div>




<script>
function updateTime() {
    var date = new Date();
    var now = date.getTime();
    var startDate = new Date("2024/07/02 09:10:00");
    var start = startDate.getTime();
    var diff = now - start;
    var y, d, h, m;
    y = Math.floor(diff / (365 * 24 * 3600 * 1000));
    diff -= y * 365 * 24 * 3600 * 1000;
    d = Math.floor(diff / (24 * 3600 * 1000));
    h = Math.floor(diff / (3600 * 1000) % 24);
    m = Math.floor(diff / (60 * 1000) % 60);
    if (y == 0) {
        document.getElementById("web-time").innerHTML = d + " 天 " + h + " 小时 " + m + " 分钟";
    } else {
        document.getElementById("web-time").innerHTML = y + " 年 " + d + " 天 " + h + " 小时 " + m + " 分钟";
    }
    setTimeout(updateTime, 1000 * 60);
}
updateTime();
function toggle_statistics() {
    var statistics = document.getElementById("statistics");
    if (statistics.style.opacity == 0) {
        statistics.style.opacity = 1;
    } else {
        statistics.style.opacity = 0;
    }
}
</script>