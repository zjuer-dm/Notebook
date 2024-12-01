## 回溯
给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

```py
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)

        ans = []
        path = [0] * n 

        def DFS(c , f):
            if c == n:
                ans.append(path.copy())
                return 
            for x in f:
                path[c] = x
                DFS(c+1 , f - {x})
        DFS(0 , set(nums))
        return ans
```