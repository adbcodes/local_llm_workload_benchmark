# Code, Debug, and Repair research notes

The benchmark uses original prompts and original test cases. No problem statement
or solution was copied from a practice platform.

## Coverage sources

The 24 DSA implementation tasks were selected by comparing the topic maps in:

- [NeetCode 150](https://neetcode.io/practice/practice/neetcode150)
- [Striver's A2Z DSA Sheet](https://takeuforward.org/dsa/strivers-a2z-sheet-learn-dsa-a-to-z)
- [LeetCode Study Plans](https://leetcode.com/studyplan/)

These sources consistently emphasize arrays and hashing, two pointers, sliding
windows, stacks, binary search, linked structures, trees, heaps, backtracking,
graphs, dynamic programming, greedy methods, intervals, tries, and bit
manipulation. The benchmark samples every major family at least once while
keeping the total practical enough to run across local models.

## Composition

| Task family | Count | Evaluation |
|---|---:|---|
| DSA function implementation | 24 | Restricted Python tests |
| Practical function implementation | 6 | Restricted Python tests |
| Bug diagnosis | 10 | Exact diagnostic label |
| Code repair | 8 | Restricted Python tests |
| **Total** | **48** | |

The implementation set is a coverage benchmark, not a claim to include every
named problem on any sheet. Prompts vary signatures, representations, edge
cases, and output contracts so memorizing one platform's wording is less useful
than understanding the underlying technique.
