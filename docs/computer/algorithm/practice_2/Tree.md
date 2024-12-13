给你一个链表的头节点 head，请你编写代码，反复删去链表中由 总和 值为 0 的连续节点组成的序列，直到不存在这样的序列为止。

删除完毕后，请你返回最终结果链表的头节点。

你可以返回任何满足题目要求的答案。

```py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeZeroSumSublists(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre = 0
        seen = {}
        dummy = ListNode(0)
        dummy.next = head
        seen[0] = dummy
        while head:
            pre += head.val
            seen[pre] = head
            head = head.next

        pre = 0
        head = dummy
        while head:
            pre += head.val
            head.next = seen[pre].next
            head = head.next
        return dummy.next
```