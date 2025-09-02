import pandas as pd
import os
import pdb
from copy import deepcopy
from constants import get_result_dir
from utils import save_json

template_dir = get_result_dir(
    dataset_name="game24",
    model_name="xai/grok-3-mini-beta",
    shot=0,
    template_type="reasoning_api",
    response_length=404,
    num_samples=100,
    feature_noise=None,
    label_noise=0.0,
    train_step=0,
    data_mode="default",
    n_query=1,
    temperature=1.0,
    replicate_id=0,
)
template = pd.read_parquet(f"{template_dir}/test_default.parquet").iloc[0].to_dict()

synthetic_data = [
    {
        "numbers": [2, 5, 6, 8],
        "reasoning": "Compute 8 / 2 = 4, leaving 2 and 3. Try 4 + 3 = 7, but the extra 2 can’t make 24, so set that aside. Still holding 8 / 2 = 4, switch to the other pair 3 * 2 = 6. Add the results 4 + 6 = 10 -> not 24. Same pair, now multiply: 4 * 6 = 24. All numbers: 2, 2, 3, 8 -> 8 / 2 = 4 -> 3 * 2 = 6 -> 4 * 6 = 24, all used once, done.",
        "answer": "8/2*3*2=24",
        "tree": {
            "node1": {
                "Problem": "2, 2, 3, 8",
                "parent": None,
                "Result": None
            },
            "node2": {
                "Problem": "8/2, 2, 3",
                "parent": "node1",
                "Result": None
            },
            "node3": {
                "Problem": "(8/2)+3, 2",
                "parent": "node2",
                "Result": None
            },
            "node4": {
                "Problem": "8/2, 3*2",
                "parent": "node2",
                "Result": None
            },
            "node5": {
                "Problem": "(8/2)+(3*2)",
                "parent": "node4",
                "Result": 10
            },
            "node6": {
                "Problem": "(8/2)*(3*2)",
                "parent": "node4",
                "Result": 24
            }
        },
        "walk": [
            {
                "from": "node1",
                "to": "node2",
                "category": "calculation/deriviation"
            },
            {
                "from": "node2",
                "to": "node3",
                "category": "calculation/deriviation"
            },
            {
                "from": "node3",
                "to": "node2",
                "category": "backtracking"
            },
            {
                "from": "node2",
                "to": "node4",
                "category": "calculation/deriviation"
            },
            {
                "from": "node4",
                "to": "node5",
                "category": "calculation/deriviation"
            },
            {
                "from": "node5",
                "to": "node4",
                "category": "backtracking"
            },
            {
                "from": "node4",
                "to": "node6",
                "category": "calculation/deriviation"
            },
            {
                "from": "node6",
                "to": "node1",
                "category": "verification"
            },
            {
                "from": "node1",
                "to": "node2",
                "category": "verification"
            },
            {
                "from": "node2",
                "to": "node4",
                "category": "verification"
            },
            {
                "from": "node4",
                "to": "node6",
                "category": "verification"
            }
        ]
    },
    {
        "numbers": [2, 2, 3, 8],
        "reasoning": "Compute 8 / 2 = 4, leaving 2 and 3. Try 4 + 3 = 7, but the extra 2 can’t make 24, so set that aside. Still holding 8 / 2 = 4, switch to the other pair 3 * 2 = 6. Add the results 4 + 6 = 10 -> not 24. Same pair, now multiply: 4 * 6 = 24. All numbers: 2, 2, 3, 8 -> 8 / 2 = 4 -> 3 * 2 = 6 -> 4 * 6 = 24, all used once, done.",
        "answer": "(8/2)*(3*2)=24",
        "tree": {
            "node1": {
                "Problem": "2, 2, 3, 8",
                "parent": None,
                "Result": None
            },
            "node2": {
                "Problem": "8/2, 2, 3",
                "parent": "node1",
                "Result": None
            },
            "node3": {
                "Problem": "(8/2)+3, 2",
                "parent": "node2",
                "Result": None
            },
            "node4": {
                "Problem": "8/2, 3*2",
                "parent": "node2",
                "Result": None
            },
            "node5": {
                "Problem": "(8/2)+(3*2)",
                "parent": "node4",
                "Result": 10
            },
            "node6": {
                "Problem": "(8/2)*(3*2)",
                "parent": "node4",
                "Result": 24
            }
        },
        "walk": [
            {
                "from": "node1",
                "to": "node2",
                "category": "calculation/deriviation"
            },
            {
                "from": "node2",
                "to": "node3",
                "category": "calculation/deriviation"
            },
            {
                "from": "node3",
                "to": "node2",
                "category": "backtracking"
            },
            {
                "from": "node2",
                "to": "node4",
                "category": "calculation/deriviation"
            },
            {
                "from": "node4",
                "to": "node5",
                "category": "calculation/deriviation"
            },
            {
                "from": "node5",
                "to": "node4",
                "category": "backtracking"
            },
            {
                "from": "node4",
                "to": "node6",
                "category": "calculation/deriviation"
            },
            {
                "from": "node6",
                "to": "node1",
                "category": "verification"
            },
            {
                "from": "node1",
                "to": "node2",
                "category": "verification"
            },
            {
                "from": "node2",
                "to": "node4",
                "category": "verification"
            },
            {
                "from": "node4",
                "to": "node6",
                "category": "verification"
            }
        ]
    },
    {
        "numbers": [3, 5, 6, 6],
        "reasoning": "I see two 6s, so my first thought is to combine them. Let's try 6 + 6 = 12. Now I need to make 2 from the remaining 5 and 3 to get to 24. That's easy, 5 - 3 = 2. So, I can just multiply the two results: 12 * 2 = 24. This looks right.",
        "answer": "(6+6)*(5-3)=24",
        "tree": {
            "node1": {
                "Problem": "3, 5, 6, 6",
                "parent": None,
                "Result": None
            },
            "node2": {
                "Problem": "6+6, 3, 5",
                "parent": "node1",
                "Result": None
            },
            "node3": {
                "Problem": "6+6, 5-3",
                "parent": "node2",
                "Result": None
            },
            "node4": {
                "Problem": "(6+6)*(5-3)",
                "parent": "node3",
                "Result": 24
            }
        },
        "walk": [
            {
                "from": "node1",
                "to": "node2",
                "category": "calculation/deriviation"
            },
            {
                "from": "node2",
                "to": "node3",
                "category": "calculation/deriviation"
            },
            {
                "from": "node3",
                "to": "node4",
                "category": "calculation/deriviation"
            },
            {
                "from": "node4",
                "to": "node1",
                "category": "verification"
            },
            {
                "from": "node1",
                "to": "node2",
                "category": "verification"
            },
            {
                "from": "node2",
                "to": "node3",
                "category": "verification"
            },
            {
                "from": "node3",
                "to": "node4",
                "category": "verification"
            }
        ]
    },
    {
        "numbers": [4, 8, 8, 10],
        "reasoning": "I see two 8s, so I can subtract them from something. Let's try making a big number first, like 10 * 4 = 40. Now I have the two 8s left. Can I get from 40 down to 24 with them? Yes, 40 - 8 is 32. And then 32 - 8 is 24. That works perfectly.",
        "answer": "(10*4)-8-8=24",
        "tree": {
            "node1": {
                "Problem": "4, 8, 8, 10",
                "parent": None,
                "Result": None
            },
            "node2": {
                "Problem": "10*4, 8, 8",
                "parent": "node1",
                "Result": None
            },
            "node3": {
                "Problem": "(10*4)-8, 8",
                "parent": "node2",
                "Result": None
            },
            "node4": {
                "Problem": "((10*4)-8)-8",
                "parent": "node3",
                "Result": 24
            }
        },
        "walk": [
            {
                "from": "node1",
                "to": "node2",
                "category": "calculation/deriviation"
            },
            {
                "from": "node2",
                "to": "node3",
                "category": "calculation/deriviation"
            },
            {
                "from": "node3",
                "to": "node4",
                "category": "calculation/deriviation"
            },
            {
                "from": "node4",
                "to": "node1",
                "category": "verification"
            },
            {
                "from": "node1",
                "to": "node2",
                "category": "verification"
            },
            {
                "from": "node2",
                "to": "node3",
                "category": "verification"
            },
            {
                "from": "node3",
                "to": "node4",
                "category": "verification"
            }
        ]
    },
    {
        "numbers": [1, 3, 4, 6],
        "reasoning": "Let's try simple multiplication. 6 * 4 = 24. Oh, that was easy, but it leaves 1 and 3 unused, so that path is invalid. I have to use all numbers. Let's backtrack. Division can make numbers bigger. To get 24 from 6, I would need to divide by 1/4. Can I make 1/4 from the other numbers, 1, 3, and 4? Let's see... 3/4 is a good start. Then 1 - 3/4 is 1/4. Yes! So the whole thing is 6 divided by (1 - 3/4). That works.",
        "answer": "6/(1-(3/4))=24",
        "tree": {
            "node1": {
                "Problem": "1, 3, 4, 6",
                "parent": None,
                "Result": None
            },
            "node2": {
                "Problem": "6*4, 1, 3",
                "parent": "node1",
                "Result": None
            },
            "node3": {
                "Problem": "3/4, 1, 6",
                "parent": "node1",
                "Result": None
            },
            "node4": {
                "Problem": "1-(3/4), 6",
                "parent": "node3",
                "Result": None
            },
            "node5": {
                "Problem": "6/(1-(3/4))",
                "parent": "node4",
                "Result": 24
            }
        },
        "walk": [
            {
                "from": "node1",
                "to": "node2",
                "category": "calculation/deriviation"
            },
            {
                "from": "node2",
                "to": "node1",
                "category": "backtracking"
            },
            {
                "from": "node1",
                "to": "node3",
                "category": "calculation/deriviation"
            },
            {
                "from": "node3",
                "to": "node4",
                "category": "calculation/deriviation"
            },
            {
                "from": "node4",
                "to": "node5",
                "category": "calculation/deriviation"
            },
            {
                "from": "node5",
                "to": "node1",
                "category": "verification"
            },
            {
                "from": "node1",
                "to": "node3",
                "category": "verification"
            },
            {
                "from": "node3",
                "to": "node4",
                "category": "verification"
            },
            {
                "from": "node4",
                "to": "node5",
                "category": "verification"
            }
        ]
    },
    {
        "numbers": [2, 3, 8, 9],
        "reasoning": "First, let's try multiplying. 9 * 8 = 72. That's big. I'm left with 2 and 3. 72 / 3 = 24. Oh, but that leaves the 2 unused, so that's not allowed. Backtrack. Okay, what if I start with 9 * 3 = 27? I have 2 and 8 left. 27-8=19. Then 19+2=21. No good. Backtrack. Let's try addition from the start. 9+8=17. Left with 2, 3. 17+3=20. Then 20+2=22. Still no. Backtrack. Maybe a different subtraction. 9-2=7. Left with 3, 8. 7*3=21. 21+8=29. No again. Backtrack. This is tricky. Let me rethink. I need to combine numbers in pairs. What about 9-3=6 and 8/2=4? That gives me 6 and 4. And 6*4=24. That works, and all numbers are used.",
        "answer": "(9-3)*(8/2)=24",
        "tree": {
            "node1": {"Problem": "2, 3, 8, 9", "parent": None, "Result": None},
            "node2": {"Problem": "9*8, 2, 3", "parent": "node1", "Result": None},
            "node3": {"Problem": "(9*8)/3, 2", "parent": "node2", "Result": None},
            "node4": {"Problem": "9*3, 2, 8", "parent": "node1", "Result": None},
            "node5": {"Problem": "(9*3)-8, 2", "parent": "node4", "Result": None},
            "node6": {"Problem": "9+8, 2, 3", "parent": "node1", "Result": None},
            "node7": {"Problem": "(9+8)+3, 2", "parent": "node6", "Result": None},
            "node8": {"Problem": "9-2, 3, 8", "parent": "node1", "Result": None},
            "node9": {"Problem": "(9-2)*3, 8", "parent": "node8", "Result": None},
            "node10": {"Problem": "9-3, 2, 8", "parent": "node1", "Result": None},
            "node11": {"Problem": "9-3, 8/2", "parent": "node10", "Result": None},
            "node12": {"Problem": "(9-3)*(8/2)", "parent": "node11", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node11", "category": "calculation/deriviation"},
            {"from": "node11", "to": "node12", "category": "calculation/deriviation"},
            {"from": "node12", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node10", "category": "verification"},
            {"from": "node10", "to": "node11", "category": "verification"},
            {"from": "node11", "to": "node12", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 4, 7, 8],
        "reasoning": "Let's start with a big multiplication, 8*7=56. I have 1 and 4. 56-4=52. 52-1=51. Not 24. Backtrack. What about 8*4=32? I have 1 and 7. I need to subtract 8. If I try 7-1=6, then 32-6=26. That's not it. Backtrack. Maybe the whole multiplication approach is wrong. Let's try building up with addition. 1+4=5. Now I have 5, 7, 8. 5+7=12. Now 12 and 8. 12+8=20. Still no. Backtrack. Okay, I need a factor of 24. Let's make one. 7-4=3. I have 1 and 8. 8*3=24. But the 1 is unused. That's an invalid path. Backtrack. Let me try making a different factor. What about 7+1=8? Now I have 4, 8, 8. Let's try 8*4=32. Now I have 32 and 8. 32-8=24. That works!",
        "answer": "(7+1)*4-8=24",
        "tree": {
            "node1": {"Problem": "1, 4, 7, 8", "parent": None, "Result": None},
            "node2": {"Problem": "8*7, 1, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "(8*7)-4, 1", "parent": "node2", "Result": None},
            "node4": {"Problem": "8*4, 1, 7", "parent": "node1", "Result": None},
            "node5": {"Problem": "(8*4)-(7-1)", "parent": "node4", "Result": 26},
            "node6": {"Problem": "1+4, 7, 8", "parent": "node1", "Result": None},
            "node7": {"Problem": "(1+4)+7, 8", "parent": "node6", "Result": None},
            "node8": {"Problem": "7-4, 1, 8", "parent": "node1", "Result": None},
            "node9": {"Problem": "(7-4)*8, 1", "parent": "node8", "Result": None},
            "node10": {"Problem": "7+1, 4, 8", "parent": "node1", "Result": None},
            "node11": {"Problem": "(7+1)*4, 8", "parent": "node10", "Result": None},
            "node12": {"Problem": "((7+1)*4)-8", "parent": "node11", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node11", "category": "calculation/deriviation"},
            {"from": "node11", "to": "node12", "category": "calculation/deriviation"},
            {"from": "node12", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node10", "category": "verification"},
            {"from": "node10", "to": "node11", "category": "verification"},
            {"from": "node11", "to": "node12", "category": "verification"}
        ]
    },
    {
        "numbers": [4, 5, 5, 9],
        "reasoning": "With two 5s, let's start with 5 * 5 = 25. Now I have 4 and 9. I need to subtract 1. But 9-4 is 5. So 25-5=20. That's a dead end. Backtrack. Let's try a different operation with the 5s: 5+5=10. Now I have 4, 9, 10. 10+9=19. Then 19+4=23. Close, but wrong. Backtrack. Okay, let's start with a different pair. 9-5=4. Now I have 4, 4, 5. Let's try 4*4=16. Then 16+5=21. No. Backtrack. What if I start with 9-4=5? Now I have 5, 5, 5. Let's try 5*5=25. Then 25-5=20. Still no. Backtrack. This is hard. Let me try a combination I haven't used yet. 5*4=20. I have 5 and 9 left. Let's add 9: 20+9=29. Now subtract 5: 29-5=24. Got it!",
        "answer": "5*4+9-5=24",
        "tree": {
            "node1": {"Problem": "4, 5, 5, 9", "parent": None, "Result": None},
            "node2": {"Problem": "5*5, 4, 9", "parent": "node1", "Result": None},
            "node3": {"Problem": "(5*5)-(9-4)", "parent": "node2", "Result": 20},
            "node4": {"Problem": "5+5, 4, 9", "parent": "node1", "Result": None},
            "node5": {"Problem": "(5+5)+9, 4", "parent": "node4", "Result": None},
            "node6": {"Problem": "9-5, 4, 5", "parent": "node1", "Result": None},
            "node7": {"Problem": "(9-5)*4, 5", "parent": "node6", "Result": None},
            "node8": {"Problem": "9-4, 5, 5", "parent": "node1", "Result": None},
            "node9": {"Problem": "(9-4)*5, 5", "parent": "node8", "Result": None},
            "node10": {"Problem": "5*4, 5, 9", "parent": "node1", "Result": None},
            "node11": {"Problem": "(5*4)+9, 5", "parent": "node10", "Result": None},
            "node12": {"Problem": "((5*4)+9)-5", "parent": "node11", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node11", "category": "calculation/deriviation"},
            {"from": "node11", "to": "node12", "category": "calculation/deriviation"},
            {"from": "node12", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node10", "category": "verification"},
            {"from": "node10", "to": "node11", "category": "verification"},
            {"from": "node11", "to": "node12", "category": "verification"}
        ]
    },
    {
        "numbers": [3, 6, 7, 9],
        "reasoning": "Let's start by multiplying. 9 * 3 = 27. I have 6 and 7. I need to subtract 3. 7-6=1. Doesn't work. Backtrack. What about 9 * 7 = 63? That's way too big. Backtrack. Let's try 6 * 7 = 42. I have 3 and 9. 42-9=33. Then 33-3=30. No. Backtrack. How about 6 * 3 = 18? I have 7 and 9. 18+7=25. Then 25-9=16. Also no. Backtrack. It seems multiplication first isn't working. Let me try subtraction first. 9-6 = 3. Now I have 3, 3, 7. Let's try multiplying from here. 3 * 7 = 21. I have a 3 left. 21 + 3 = 24. It worked!",
        "answer": "(9-6)*7+3=24",
        "tree": {
            "node1": {"Problem": "3, 6, 7, 9", "parent": None, "Result": None},
            "node2": {"Problem": "9*3, 6, 7", "parent": "node1", "Result": None},
            "node3": {"Problem": "(9*3)-7, 6", "parent": "node2", "Result": None},
            "node4": {"Problem": "9*7, 3, 6", "parent": "node1", "Result": None},
            "node5": {"Problem": "6*7, 3, 9", "parent": "node1", "Result": None},
            "node6": {"Problem": "(6*7)-9, 3", "parent": "node5", "Result": None},
            "node7": {"Problem": "6*3, 7, 9", "parent": "node1", "Result": None},
            "node8": {"Problem": "(6*3)+7, 9", "parent": "node7", "Result": None},
            "node9": {"Problem": "9-6, 3, 7", "parent": "node1", "Result": None},
            "node10": {"Problem": "(9-6)*7, 3", "parent": "node9", "Result": None},
            "node11": {"Problem": "((9-6)*7)+3", "parent": "node10", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node11", "category": "calculation/deriviation"},
            {"from": "node11", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"},
            {"from": "node10", "to": "node11", "category": "verification"}
        ]
    },
    {
        "numbers": [2, 2, 6, 7],
        "reasoning": "Let's try multiplying the big numbers, 7 * 6 = 42. Too big. Backtrack. How about 6 * 2 = 12? I have 2 and 7 left. Let's try 12*2=24. But that leaves the 7 unused, which is invalid. Backtrack. Let's go back to having 12, 2, 7. Let's try addition. 12+7=19. Then 19+2=21. Not 24. Backtrack. Okay, let's start with a different multiplication: 7*2=14. I have 2 and 6. 14+6=20. Then 20+2=22. Still no. Backtrack. There are a lot of ways to get close. Maybe I should add first. 7+2=9. Now I have 2, 6, 9. Let's try 9*2=18. And I have a 6 left. 18+6=24. That's the solution.",
        "answer": "(7+2)*2+6=24",
        "tree": {
            "node1": {"Problem": "2, 2, 6, 7", "parent": None, "Result": None},
            "node2": {"Problem": "7*6, 2, 2", "parent": "node1", "Result": None},
            "node3": {"Problem": "6*2, 2, 7", "parent": "node1", "Result": None},
            "node4": {"Problem": "(6*2)*2, 7", "parent": "node3", "Result": None},
            "node5": {"Problem": "(6*2)+7, 2", "parent": "node3", "Result": None},
            "node6": {"Problem": "7*2, 2, 6", "parent": "node1", "Result": None},
            "node7": {"Problem": "(7*2)+6, 2", "parent": "node6", "Result": None},
            "node8": {"Problem": "7+2, 2, 6", "parent": "node1", "Result": None},
            "node9": {"Problem": "(7+2)*2, 6", "parent": "node8", "Result": None},
            "node10": {"Problem": "((7+2)*2)+6", "parent": "node9", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"}
        ]
    },
    {
        "numbers": [3, 3, 4, 8],
        "reasoning": "My first thought is always to try the direct multiplication: 8 * 3 = 24. But this leaves the other 3 and the 4 unused. That's an invalid path. Backtrack. Let's try a different multiplication: 4 * 3 = 12. Now I have 3 and 8 left. 12 * 3 = 36. Then 36-8=28. No. Backtrack. What if I combine the identical numbers first? 3+3=6. I'm left with 4 and 8. Let's try 8*4=32. Then 32-6=26. Still no. Backtrack. Maybe division is the key. 8/4=2. Now I have 2, 3, 3. 2+3=5. Then 5+3=8. Not a solution. Backtrack. This is a hard one. I need to make 3 to go with the 8. Can I make 3 from 3, 3, 4? I can use division to make a small number first, like 3/3=1. Now I have 1, 4, 8. Can I make 3 from 4 and 1? Yes, 4-1=3. So the full expression is 8 * (4 - 3/3). That works!",
        "answer": "8*(4-3/3)=24",
        "tree": {
            "node1": {"Problem": "3, 3, 4, 8", "parent": None, "Result": None},
            "node2": {"Problem": "8*3, 3, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "4*3, 3, 8", "parent": "node1", "Result": None},
            "node4": {"Problem": "(4*3)*3, 8", "parent": "node3", "Result": None},
            "node5": {"Problem": "3+3, 4, 8", "parent": "node1", "Result": None},
            "node6": {"Problem": "8*4, 6", "parent": "node5", "Result": None},
            "node7": {"Problem": "8/4, 3, 3", "parent": "node1", "Result": None},
            "node8": {"Problem": "(8/4)+3, 3", "parent": "node7", "Result": None},
            "node9": {"Problem": "3/3, 4, 8", "parent": "node1", "Result": None},
            "node10": {"Problem": "4-(3/3), 8", "parent": "node9", "Result": None},
            "node11": {"Problem": "8*(4-(3/3))", "parent": "node10", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node11", "category": "calculation/deriviation"},
            {"from": "node11", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"},
            {"from": "node10", "to": "node11", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 6, 7, 9],
        "reasoning": "Let's try some large multiplications first. 9 * 7 = 63. That's too big to come back from. Backtrack. How about 9 * 6 = 54? Still too big. Backtrack. Okay, what about 7 * 6 = 42? To get to 24, I need to subtract 18. I can't make 18 from 1 and 9. Backtrack. Maybe the answer is just simple addition? 9+7+6+1 = 23. That's very close, but not 24. Backtrack again. This suggests a more complex structure. Maybe I need to partition the numbers and combine the results. I need two numbers that multiply to 24, like 3 and 8. Can I make 3? Yes, 9-6=3. Can I make 8 from the remaining 1 and 7? Yes, 7+1=8. So, (9-6)*(7+1) = 24. That's the solution.",
        "answer": "(9-6)*(7+1)=24",
        "tree": {
            "node1": {"Problem": "1, 6, 7, 9", "parent": None, "Result": None},
            "node2": {"Problem": "9*7, 1, 6", "parent": "node1", "Result": None},
            "node3": {"Problem": "9*6, 1, 7", "parent": "node1", "Result": None},
            "node4": {"Problem": "7*6, 1, 9", "parent": "node1", "Result": None},
            "node5": {"Problem": "9+7, 1, 6", "parent": "node1", "Result": None},
            "node6": {"Problem": "(9+7)+6, 1", "parent": "node5", "Result": None},
            "node7": {"Problem": "9-6, 1, 7", "parent": "node1", "Result": None},
            "node8": {"Problem": "9-6, 7+1", "parent": "node7", "Result": None},
            "node9": {"Problem": "(9-6)*(7+1)", "parent": "node8", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node7", "category": "verification"},
            {"from": "node7", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"}
        ]
    },
    {
        "numbers": [3, 4, 5, 8],
        "reasoning": "Let's start with a big multiplication, 8 * 5 = 40. That leaves 3 and 4. 40-4=36, then 36-3=33. No. Backtrack to the start. Let's try making a smaller number first. 5-3=2. This seems like a promising first step. Now I have the expression (5-3), and the numbers 4 and 8. From here, let's try combining the larger numbers, 8*4=32. Now I have the expressions (8*4) and (5-3). That's not going to make 24. Okay, that sub-path was a dead end, but I still think (5-3) was a good start. Let me backtrack just to that point. So I have (5-3), 4, and 8 again. Instead of multiplying the 8 and 4, what if I add them? That gives me the expression (4+8). Now I have two expressions, (5-3) and (4+8). Let's multiply them. 2 * 12 = 24. Success!",
        "answer": "(5-3)*(4+8)=24",
        "tree": {
            "node1": {"Problem": "3, 4, 5, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(8*5), 3, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "((8*5)-4), 3", "parent": "node2", "Result": None},
            "node4": {"Problem": "(5-3), 4, 8", "parent": "node1", "Result": None},
            "node5": {"Problem": "(8*4), (5-3)", "parent": "node4", "Result": None},
            "node6": {"Problem": "((8*4)-(5-3))", "parent": "node5", "Result": 30},
            "node7": {"Problem": "(4+8), (5-3)", "parent": "node4", "Result": None},
            "node8": {"Problem": "((5-3)*(4+8))", "parent": "node7", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node4", "category": "backtracking"},
            {"from": "node4", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node4", "category": "verification"},
            {"from": "node4", "to": "node7", "category": "verification"},
            {"from": "node7", "to": "node8", "category": "verification"}
        ]
    },
    {
        "numbers": [2, 6, 6, 8],
        "reasoning": "I see two 6s, so maybe 6*6=36. I have 2 and 8 left. 36-8=28. Then 28-2=26. Not a solution. Backtrack to the start. Let's try a different multiplication, 8*6=48. Left with 2 and 6. 48-6=42. 42-2=40. Still no good. Backtrack to the start again. Okay, those big multiplications didn't work. Let's try to make a very simple number, like 1. I can do 6/6=1. Now I have the expression (6/6), and the numbers 2 and 8. This feels like a useful step. From here, let's try multiplying the remaining big numbers, 8*2=16. Now I have the expressions (8*2) and (6/6). Adding them gives 17, not 24. Okay, that sub-path failed, but the (6/6) idea is still good. Let's backtrack just to that point. I have (6/6), 2, and 8 again. Instead of multiplying, let me add the 2 to my (6/6) expression. This gives me ((6/6)+2). Now I have that expression and the number 8. 3*8=24. That's the one!",
        "answer": "((6/6)+2)*8=24",
        "tree": {
            "node1": {"Problem": "2, 6, 6, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(6*6), 2, 8", "parent": "node1", "Result": None},
            "node3": {"Problem": "((6*6)-8), 2", "parent": "node2", "Result": None},
            "node4": {"Problem": "(8*6), 2, 6", "parent": "node1", "Result": None},
            "node5": {"Problem": "((8*6)-6), 2", "parent": "node4", "Result": None},
            "node6": {"Problem": "(6/6), 2, 8", "parent": "node1", "Result": None},
            "node7": {"Problem": "(8*2), (6/6)", "parent": "node6", "Result": None},
            "node8": {"Problem": "((8*2)+(6/6))", "parent": "node7", "Result": 17},
            "node9": {"Problem": "((6/6)+2), 8", "parent": "node6", "Result": None},
            "node10": {"Problem": "(((6/6)+2)*8)", "parent": "node9", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node6", "category": "backtracking"},
            {"from": "node6", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node6", "category": "verification"},
            {"from": "node6", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 6, 7, 9],
        "reasoning": "Let's try multiplying large numbers first. 9 * 7 = 63. That's too large to work. Backtrack. How about 7 * 6 = 42? To get 24, I need to subtract 18. I can't make 18 from 1 and 9. Backtrack. My next thought is simple addition: 9+7=16. Then 16+6=22. And 22+1=23. So close, but that's a dead end. Backtrack. Okay, what if I start with a subtraction? 9-1=8. That seems promising. Now I have 6, 7, 8. Let's try multiplying from here: 8*7=56. Still too big. This sub-path is wrong, but maybe 9-1=8 was a good start. Let's backtrack to that point. I have 6, 7, 8 again. Instead of multiplying, what if I add? 6+7=13. 13+8=21. Still not right. This means 9-1=8 was not the right first move after all. Let me backtrack to the beginning. The solution must be a partition. I need factors of 24, like 3 and 8. Can I make 3? 9-6=3. Yes. Can I make 8 with the remaining 1 and 7? 1+7=8. Yes. So, (9-6)*(1+7) must be the answer.",
        "answer": "(9-6)*(1+7)=24",
        "tree": {
            "node1": {"Problem": "1, 6, 7, 9", "parent": None, "Result": None},
            "node2": {"Problem": "(9*7), 1, 6", "parent": "node1", "Result": None},
            "node3": {"Problem": "(7*6), 1, 9", "parent": "node1", "Result": None},
            "node4": {"Problem": "(9+7), 1, 6", "parent": "node1", "Result": None},
            "node5": {"Problem": "((9+7)+6), 1", "parent": "node4", "Result": None},
            "node6": {"Problem": "(9-1), 6, 7", "parent": "node1", "Result": None},
            "node7": {"Problem": "((9-1)*7), 6", "parent": "node6", "Result": None},
            "node8": {"Problem": "((9-1)+6), 7", "parent": "node6", "Result": None},
            "node9": {"Problem": "(9-6), 1, 7", "parent": "node1", "Result": None},
            "node10": {"Problem": "(9-6), (1+7)", "parent": "node9", "Result": None},
            "node11": {"Problem": "((9-6)*(1+7))", "parent": "node10", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node6", "category": "backtracking"},
            {"from": "node6", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node11", "category": "calculation/deriviation"},
            {"from": "node11", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"},
            {"from": "node10", "to": "node11", "category": "verification"}
        ]
    },
    {
        "numbers": [3, 4, 7, 7],
        "reasoning": "With two 7s, my first move is 7*7=49. That's too big. Backtrack. What about 7*4=28? I need to subtract 4. I can't make 4 from 3 and 7. Backtrack. Let's try 7*3=21. I need to add 3. I can't make 3 from 4 and 7. Backtrack. Let's try subtraction first: 7-4=3. Now I have 3, 3, 7. This seems promising. Let me try multiplying the identical numbers: 3*3=9. Now I have 7 and 9. 7+9=16. That's not a solution. Okay, the 3*3 part was wrong, but maybe the 7-4=3 start was right. Let's backtrack to that point. I have 3, 3, 7 again. Instead of 3*3, let's try 3*7=21. Now I have 3 and 21. And 21+3=24. That works!",
        "answer": "(7-4)*7+3=24",
        "tree": {
            "node1": {"Problem": "3, 4, 7, 7", "parent": None, "Result": None},
            "node2": {"Problem": "(7*7), 3, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "(7*4), 3, 7", "parent": "node1", "Result": None},
            "node4": {"Problem": "(7*3), 4, 7", "parent": "node1", "Result": None},
            "node5": {"Problem": "(7-4), 3, 7", "parent": "node1", "Result": None},
            "node6": {"Problem": "((7-4)*3), 7", "parent": "node5", "Result": None},
            "node7": {"Problem": "(((7-4)*3)+7)", "parent": "node6", "Result": 16},
            "node8": {"Problem": "((7-4)*7), 3", "parent": "node5", "Result": None},
            "node9": {"Problem": "(((7-4)*7)+3)", "parent": "node8", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node5", "category": "verification"},
            {"from": "node5", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"}
        ]
    },
    {
        "numbers": [2, 3, 5, 6],
        "reasoning": "Let's start by multiplying 6*5=30. Now I have 2 and 3. I need to subtract 6. I can't get 6 from 3-2=1 or 3+2=5. Backtrack. What about 6*3=18? Now I have 2 and 5. 18+2=20. 20+5=25. No. Backtrack. Let's try 5*3=15. I have 2 and 6. 15+6=21. 21+2=23. Still no. Backtrack. What about 6*2=12? I have 3 and 5. 12+3=15. 15+5=20. Backtrack again. It seems like a single chain isn't working. I need to make two separate parts. Let's go back to that first idea. 6*5=30. And the other pair is 2 and 3. I can make 6 from 2*3. Now I have two results: 30 and 6. 30-6=24. That's the solution.",
        "answer": "(6*5)-(3*2)=24",
        "tree": {
            "node1": {"Problem": "2, 3, 5, 6", "parent": None, "Result": None},
            "node2": {"Problem": "(6*5), 2, 3", "parent": "node1", "Result": None},
            "node3": {"Problem": "(6*3), 2, 5", "parent": "node1", "Result": None},
            "node4": {"Problem": "((6*3)+2), 5", "parent": "node3", "Result": None},
            "node5": {"Problem": "(5*3), 2, 6", "parent": "node1", "Result": None},
            "node6": {"Problem": "((5*3)+6), 2", "parent": "node5", "Result": None},
            "node7": {"Problem": "(6*2), 3, 5", "parent": "node1", "Result": None},
            "node8": {"Problem": "((6*2)+3), 5", "parent": "node7", "Result": None},
            "node9": {"Problem": "(6*5), (3*2)", "parent": "node2", "Result": None},
            "node10": {"Problem": "((6*5)-(3*2))", "parent": "node9", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"}
        ]
    },
    {
        "numbers": [4, 4, 5, 7],
        "reasoning": "With two 4s, I'll start with 4*4=16. Now I have 5 and 7. 16+5=21. 21+7=28. No. Backtrack. What about 5*4=20? I have 4 and 7. 20+7=27. 27-4=23. No. Backtrack. Let's try subtraction. 7-5=2. That gives me 2, 4, 4. This is a good intermediate step. Let's explore from here. 4*4=16. Now I have 2 and 16. 2+16=18. Dead end. Okay, 7-5=2 was a good start, let's backtrack to that point. I have 2, 4, 4 again. Instead of 4*4, what if I do 4+4=8? Now I have 2 and 8. That gives 10 or 16, not 24. This sub-path is also wrong. Backtrack to the 2, 4, 4 state one more time. Let's try 2+4=6. Now I have 4 and 6. And 4*6=24! That was a deep one.",
        "answer": "(7-5+4)*4=24",
        "tree": {
            "node1": {"Problem": "4, 4, 5, 7", "parent": None, "Result": None},
            "node2": {"Problem": "(4*4), 5, 7", "parent": "node1", "Result": None},
            "node3": {"Problem": "((4*4)+5), 7", "parent": "node2", "Result": None},
            "node4": {"Problem": "(5*4), 4, 7", "parent": "node1", "Result": None},
            "node5": {"Problem": "((5*4)+7), 4", "parent": "node4", "Result": None},
            "node6": {"Problem": "(7-5), 4, 4", "parent": "node1", "Result": None},
            "node7": {"Problem": "(4*4), (7-5)", "parent": "node6", "Result": None},
            "node8": {"Problem": "(4+4), (7-5)", "parent": "node6", "Result": None},
            "node9": {"Problem": "((7-5)+4), 4", "parent": "node6", "Result": None},
            "node10": {"Problem": "(((7-5)+4)*4)", "parent": "node9", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node6", "category": "backtracking"},
            {"from": "node6", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node6", "category": "backtracking"},
            {"from": "node6", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node6", "category": "verification"},
            {"from": "node6", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"}
        ]
    },
    {
        "numbers": [4, 6, 6, 9],
        "reasoning": "With two 6s, my first try is 6*6=36. I have 4 and 9 left. 36-9=27. 27-4=23. Close. Backtrack. Let's try 9*6=54. That's too large. Backtrack. How about 9*4=36? Now I need to subtract 12. I can do that with 6+6=12. So, 9*4-(6+6) = 24. That's a solution. But let's say I don't see the partition. Let's say from 9*4=36, I try subtracting one 6 to get 30, then the other 6 to get 24. So 9*4-6-6=24. That's another solution. Let's explore other paths first for the example. What if I start with 6+6=12? Now I have 4, 9, 12. Let's try 12*4=48. Then 48-9=39. No. Backtrack to the 4,9,12 state. From there, let's try 12+9=21. Then 21-4=17. No. Backtrack to the start. The path 9*4=36, then subtracting the 6s one by one, seems like a good reasoning path.",
        "answer": "9*4-6-6=24",
        "tree": {
            "node1": {"Problem": "4, 6, 6, 9", "parent": None, "Result": None},
            "node2": {"Problem": "(6*6), 4, 9", "parent": "node1", "Result": None},
            "node3": {"Problem": "((6*6)-9), 4", "parent": "node2", "Result": None},
            "node4": {"Problem": "(9*6), 4, 6", "parent": "node1", "Result": None},
            "node5": {"Problem": "(6+6), 4, 9", "parent": "node1", "Result": None},
            "node6": {"Problem": "((6+6)*4), 9", "parent": "node5", "Result": None},
            "node7": {"Problem": "((6+6)+9), 4", "parent": "node5", "Result": None},
            "node8": {"Problem": "(9*4), 6, 6", "parent": "node1", "Result": None},
            "node9": {"Problem": "((9*4)-6), 6", "parent": "node8", "Result": None},
            "node10": {"Problem": "(((9*4)-6)-6)", "parent": "node9", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"}
        ]
    },
    {
        "numbers": [2, 5, 6, 8],
        "reasoning": "Let's try multiplying the biggest numbers, 8 * 6 = 48. Now I have 2 and 5. 48-5=43, and 43-2=41. That's a dead end. Backtrack. What about 8 * 5 = 40? I'm left with 2 and 6. I need to subtract 16. I can't make 16 from 2 and 6. Backtrack again. Okay, maybe I need to start with subtraction. 8-2=6. Now I have 5, 6, 6. This seems promising. Let me try multiplying the two 6s, 6*6=36. Now I have 5 and 36. 36-5=31. Not a solution. Okay, the 6*6 part was wrong, but the initial 8-2=6 might be good. Let's backtrack to having 5, 6, 6. Instead of multiplying, let's try adding: 6+6=12. Now I have 5 and 12. 12+5=17. Still no. This whole branch seems wrong. Let me backtrack to the start. The solution must involve partitioning. Let's try 5*6=30 and 8-2=6. Now I have two results, 30 and 6. 30-6=24. That works!",
        "answer": "(5*6)-(8-2)=24",
        "tree": {
            "node1": {"Problem": "2, 5, 6, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(8*6), 2, 5", "parent": "node1", "Result": None},
            "node3": {"Problem": "((8*6)-5), 2", "parent": "node2", "Result": None},
            "node4": {"Problem": "(8*5), 2, 6", "parent": "node1", "Result": None},
            "node5": {"Problem": "(8-2), 5, 6", "parent": "node1", "Result": None},
            "node6": {"Problem": "(6*6), 5", "parent": "node5", "Result": None},
            "node7": {"Problem": "(6+6), 5", "parent": "node5", "Result": None},
            "node8": {"Problem": "(5*6), 2, 8", "parent": "node1", "Result": None},
            "node9": {"Problem": "(5*6), (8-2)", "parent": "node8", "Result": None},
            "node10": {"Problem": "((5*6)-(8-2))", "parent": "node9", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"}
        ]
    },
    {
        "numbers": [4, 4, 6, 8],
        "reasoning": "Let's try multiplying the big numbers, 8*6=48. I have two 4s left. 48-4=44, 44-4=40. No. Backtrack. How about 6*4=24? That hits the target, but leaves 4 and 8 unused, so it's invalid. Backtrack. Let's try starting with subtraction. 8-4=4. Now I have 4, 4, 6. This seems like a good intermediate state. From here, let's try 4*4=16. Now I have 6 and 16. 16+6=22. Not 24. This sub-path is a dead end. I'll backtrack to the state with 4, 4, 6. From here, let's try 6*4=24. Again, this leaves a 4 unused, so it's not valid. Backtrack to the start. The solution must be a partition. I need two factors, like 12 and 2. Can I make 12? Yes, 8+4. Can I make 2 from the remaining 4 and 6? Yes, 6-4. So, (8+4)*(6-4) is the answer.",
        "answer": "(8+4)*(6-4)=24",
        "tree": {
            "node1": {"Problem": "4, 4, 6, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(8*6), 4, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "(6*4), 4, 8", "parent": "node1", "Result": None},
            "node4": {"Problem": "(8-4), 4, 6", "parent": "node1", "Result": None},
            "node5": {"Problem": "(4*4), 6", "parent": "node4", "Result": None},
            "node6": {"Problem": "(6*4), 4", "parent": "node4", "Result": None},
            "node7": {"Problem": "(8+4), 4, 6", "parent": "node1", "Result": None},
            "node8": {"Problem": "(8+4), (6-4)", "parent": "node7", "Result": None},
            "node9": {"Problem": "((8+4)*(6-4))", "parent": "node8", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node4", "category": "backtracking"},
            {"from": "node4", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node7", "category": "verification"},
            {"from": "node7", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"}
        ]
    },
    {
        "numbers": [3, 4, 6, 9],
        "reasoning": "Let's try a big multiplication first. 9*6=54. Too large. Backtrack. How about 9*4=36? I need to subtract 12. I can't make 12 from 3 and 6. Backtrack. Let's try subtraction: 9-6=3. Now I have 3, 3, 4. Let's explore this state. Try 3*3=9. Now I have 4 and 9. 9+4=13. No. Let's backtrack to the 3,3,4 state. Let's try 3*4=12 instead. Now I have 3 and 12. 12+3=15. Still no. Backtrack to the start. The linear combination must be the key. Let's try 9+3=12. Now I have 4, 6, 12. Then 12-6=6. Now I have 4 and 6. And 6*4=24. That works!",
        "answer": "((9+3)-6)*4=24",
        "tree": {
            "node1": {"Problem": "3, 4, 6, 9", "parent": None, "Result": None},
            "node2": {"Problem": "(9*6), 3, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "(9*4), 3, 6", "parent": "node1", "Result": None},
            "node4": {"Problem": "(9-6), 3, 4", "parent": "node1", "Result": None},
            "node5": {"Problem": "((9-6)*3), 4", "parent": "node4", "Result": None},
            "node6": {"Problem": "((9-6)*4), 3", "parent": "node4", "Result": None},
            "node7": {"Problem": "(9+3), 4, 6", "parent": "node1", "Result": None},
            "node8": {"Problem": "((9+3)-6), 4", "parent": "node7", "Result": None},
            "node9": {"Problem": "(((9+3)-6)*4)", "parent": "node8", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node4", "category": "backtracking"},
            {"from": "node4", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node7", "category": "verification"},
            {"from": "node7", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"}
        ]
    },
    {
        "numbers": [2, 4, 10, 12],
        "reasoning": "Let's try 12*2=24. This is an invalid solution as it leaves 4 and 10 unused. Backtrack. What about 12*4=48? I need to subtract 24. I can't make 24 from 2 and 10. Backtrack. What about 10*4=40? I need to subtract 16. I can't make 16 from 2 and 12. Backtrack. Okay, the big multiplications aren't working. Let's try subtraction. 12-10=2. Now I have 2, 2, 4. This is a good intermediate state. Let me try 2*2=4. Now I have 4 and 4. 4+4=8. Not 24. Let me backtrack to having 2, 2, 4. What if I try 4*2=8? Then I have 8 and the other 2. 8+2=10. Still no. Backtrack to the start. Let's try a different first move. 10-4=6. Now I have 2, 6, 12. From here, 6*2=12. And now I have 12 and 12. 12+12=24. Got it.",
        "answer": "(10-4)*2+12=24",
        "tree": {
            "node1": {"Problem": "2, 4, 10, 12", "parent": None, "Result": None},
            "node2": {"Problem": "(12*2), 4, 10", "parent": "node1", "Result": None},
            "node3": {"Problem": "(12*4), 2, 10", "parent": "node1", "Result": None},
            "node4": {"Problem": "(10*4), 2, 12", "parent": "node1", "Result": None},
            "node5": {"Problem": "(12-10), 2, 4", "parent": "node1", "Result": None},
            "node6": {"Problem": "((12-10)*2), 4", "parent": "node5", "Result": None},
            "node7": {"Problem": "((12-10)*4), 2", "parent": "node5", "Result": None},
            "node8": {"Problem": "(10-4), 2, 12", "parent": "node1", "Result": None},
            "node9": {"Problem": "((10-4)*2), 12", "parent": "node8", "Result": None},
            "node10": {"Problem": "(((10-4)*2)+12)", "parent": "node9", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"}
        ]
    },
    {
        "numbers": [4, 5, 7, 9],
        "reasoning": "Let's try to multiply the big numbers. 9 * 7 = 63. Too large. Backtrack. What about 9 * 5 = 45? I need to subtract 21. I can't make 21 from 4 and 7. Backtrack. What about 9 * 4 = 36? I need to subtract 12. I can't make 12 from 5 and 7. Backtrack. Let's try subtraction as a first step. 9-7=2. This gives me 2, 4, 5. This is a good intermediate step to explore. From here, let's try 5*4=20. Now I have 2 and 20. That gives 22. Not a solution. I'll backtrack to the 2,4,5 state. What if I try 5-4=1? Now I have 1 and 2. That gives 3. Still no good. Backtrack to the start. All those paths failed. Let me try a different first move. 9-5=4. Now I have 4, 4, 7. Let's try 4*7=28. Now I have 4 and 28. And 28-4=24. Found it!",
        "answer": "(9-5)*7-4=24",
        "tree": {
            "node1": {"Problem": "4, 5, 7, 9", "parent": None, "Result": None},
            "node2": {"Problem": "(9*7), 4, 5", "parent": "node1", "Result": None},
            "node3": {"Problem": "(9*5), 4, 7", "parent": "node1", "Result": None},
            "node4": {"Problem": "(9*4), 5, 7", "parent": "node1", "Result": None},
            "node5": {"Problem": "(9-7), 4, 5", "parent": "node1", "Result": None},
            "node6": {"Problem": "((9-7)*5), 4", "parent": "node5", "Result": None},
            "node7": {"Problem": "((9-7)*4), 5", "parent": "node5", "Result": None},
            "node8": {"Problem": "(9-5), 4, 7", "parent": "node1", "Result": None},
            "node9": {"Problem": "((9-5)*7), 4", "parent": "node8", "Result": None},
            "node10": {"Problem": "(((9-5)*7)-4)", "parent": "node9", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"}
        ]
    },
    {
        "numbers": [4, 5, 7, 9],
        "reasoning": "Let's try to multiply the big numbers. 9 * 7 = 63. Too large. Backtrack. What about 9 * 5 = 45? I need to subtract 21. I can't make 21 from 4 and 7. Backtrack. What about 9 * 4 = 36? I need to subtract 12. I can't make 12 from 5 and 7. Backtrack. Let's try subtraction as a first step. 9-7=2. This gives me 2, 4, 5. This is a good intermediate step to explore. From here, let's try 5*4=20. Now I have 2 and 20. That gives 22. Not a solution. I'll backtrack to the 2,4,5 state. What if I try 5-4=1? Now I have 1 and 2. That gives 3. Still no good. Backtrack to the start. All those paths failed. Let me try a different first move. 9-5=4. Now I have 4, 4, 7. Let's try 4*7=28. Now I have 4 and 28. And 28-4=24. Found it!",
        "answer": "(9-5)*7-4=24",
        "tree": {
            "node1": {"Problem": "4, 5, 7, 9", "parent": None, "Result": None},
            "node2": {"Problem": "(9*7), 4, 5", "parent": "node1", "Result": None},
            "node3": {"Problem": "(9*5), 4, 7", "parent": "node1", "Result": None},
            "node4": {"Problem": "(9*4), 5, 7", "parent": "node1", "Result": None},
            "node5": {"Problem": "(9-7), 4, 5", "parent": "node1", "Result": None},
            "node6": {"Problem": "((9-7)*5), 4", "parent": "node5", "Result": None},
            "node7": {"Problem": "((9-7)*4), 5", "parent": "node5", "Result": None},
            "node8": {"Problem": "(9-5), 4, 7", "parent": "node1", "Result": None},
            "node9": {"Problem": "((9-5)*7), 4", "parent": "node8", "Result": None},
            "node10": {"Problem": "(((9-5)*7)-4)", "parent": "node9", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"}
        ]
    },
    {
        "numbers": [4, 6, 6, 9],
        "reasoning": "With two 6s, 6*6=36 seems like a good start. I have 4 and 9 left. 36-9=27, then 27-4=23. Close, but a dead end. Backtrack. What about 9*6=54? Too large. Backtrack. Let's try starting with subtraction. 9-6=3. Now I have 3, 4, and the other 6. This seems like a promising intermediate step. From here, let's try 6*4=24. But that leaves the 3 unused, which is invalid. Backtrack to the state with 3, 4, 6. Let's try 6*3=18 instead. Now I have 4 and 18. 18+4=22. Still no good. This entire branch starting with 9-6 seems wrong. Let me backtrack to the beginning. Let's reconsider 9*4=36. I need to subtract 12. Can I make 12 from the two 6s? Yes, 6+6=12. So, the solution is 9*4 - (6+6).",
        "answer": "(9*4)-(6+6)=24",
        "tree": {
            "node1": {"Problem": "4, 6, 6, 9", "parent": None, "Result": None},
            "node2": {"Problem": "(6*6), 4, 9", "parent": "node1", "Result": None},
            "node3": {"Problem": "((6*6)-9), 4", "parent": "node2", "Result": None},
            "node4": {"Problem": "(9*6), 4, 6", "parent": "node1", "Result": None},
            "node5": {"Problem": "(9-6), 4, 6", "parent": "node1", "Result": None},
            "node6": {"Problem": "((9-6)*4), 6", "parent": "node5", "Result": None},
            "node7": {"Problem": "((9-6)*6), 4", "parent": "node5", "Result": None},
            "node8": {"Problem": "(9*4), 6, 6", "parent": "node1", "Result": None},
            "node9": {"Problem": "(9*4), (6+6)", "parent": "node8", "Result": None},
            "node10": {"Problem": "((9*4)-(6+6))", "parent": "node9", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 5, 8, 9],
        "reasoning": "Let's try multiplying the large numbers, 9 * 8 = 72. That's too big. Backtrack. How about 9 * 5 = 45? To get 24, I need to subtract 21. I can't make 21 from 1 and 8. Backtrack. Okay, what about 8 * 5 = 40? I need to subtract 16. I can't make 16 from 1 and 9. Backtrack. This suggests I need to create smaller numbers first. Let's try 9-1=8. Now I have 5, 8, 8. This looks like a promising intermediate state. From here, let's try multiplying the two 8s: 8*8=64. Too big. Let's backtrack to the 5, 8, 8 state. What if I try 8+8=16? Now I have 5 and 16. That can't make 24. This whole branch is a dead end. Let me backtrack to the start and try a different partition. I need two factors of 24, like 8 and 3. I can make 8 with 9-1. I can make 3 with 8-5. So, (9-1)*(8-5) is the answer.",
        "answer": "(9-1)*(8-5)=24",
        "tree": {
            "node1": {"Problem": "1, 5, 8, 9", "parent": None, "Result": None},
            "node2": {"Problem": "(9*8), 1, 5", "parent": "node1", "Result": None},
            "node3": {"Problem": "(9*5), 1, 8", "parent": "node1", "Result": None},
            "node4": {"Problem": "(8*5), 1, 9", "parent": "node1", "Result": None},
            "node5": {"Problem": "(9-1), 5, 8", "parent": "node1", "Result": None},
            "node6": {"Problem": "((9-1)*8), 5", "parent": "node5", "Result": None},
            "node7": {"Problem": "((9-1)+8), 5", "parent": "node5", "Result": None},
            "node8": {"Problem": "(9-1), (8-5)", "parent": "node5", "Result": None},
            "node9": {"Problem": "((9-1)*(8-5))", "parent": "node8", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node5", "category": "verification"},
            {"from": "node5", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"}
        ]
    },
    {
        "numbers": [3, 6, 9, 12],
        "reasoning": "Let's try multiplying the large numbers. 12 * 9 = 108. Too big. Backtrack. What about 12 * 6 = 72? I have 3 and 9 left. I need to divide by 3. 9-3=6, not 3. This doesn't seem easy. Backtrack. Okay, let's try division first. 12/6=2. Now I have 2, 3, 9. This seems like a promising intermediate step. From here, let's try 9*3=27. Now I have 2 and 27. No way to make 24. Let's backtrack to the 2,3,9 state. How about 9+3=12? Now I have 2 and 12. And 2*12=24. This is a valid solution! It seems that starting with division was the right idea, I just picked the wrong follow-up move the first time.",
        "answer": "(12/6)*(9+3)=24",
        "tree": {
            "node1": {"Problem": "3, 6, 9, 12", "parent": None, "Result": None},
            "node2": {"Problem": "(12*9), 3, 6", "parent": "node1", "Result": None},
            "node3": {"Problem": "(12*6), 3, 9", "parent": "node1", "Result": None},
            "node4": {"Problem": "(12/6), 3, 9", "parent": "node1", "Result": None},
            "node5": {"Problem": "((12/6)*9), 3", "parent": "node4", "Result": None},
            "node6": {"Problem": "((12/6)*3), 9", "parent": "node4", "Result": None},
            "node7": {"Problem": "(12/6), (9+3)", "parent": "node4", "Result": None},
            "node8": {"Problem": "((12/6)*(9+3))", "parent": "node7", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node4", "category": "backtracking"},
            {"from": "node4", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node4", "category": "backtracking"},
            {"from": "node4", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node4", "category": "verification"},
            {"from": "node4", "to": "node7", "category": "verification"},
            {"from": "node7", "to": "node8", "category": "verification"}
        ]
    },
    {
        "numbers": [4, 5, 6, 7],
        "reasoning": "Let's try multiplying the large numbers. 7*6=42. Too big. Backtrack. How about 7*5=35? Need to subtract 11. I can't make 11 from 4 and 6. Backtrack. What about 6*5=30? Need to subtract 6. I can get 6 from 7-4=3. No, that's 3. Backtrack. Let's try subtraction first. 7-4=3. Now I have 3, 5, 6. This is a good intermediate state. Let's try 6*5=30 from here. 30-3=27. Not a solution. Let's backtrack to the 3,5,6 state. What if I try 6-5=1? Now I have 1 and 3. 1+3=4. No. Backtrack to the start. The solution must be a partition. I need factors of 24, like 2 and 12. Can I make 2? Yes, 6-4. Can I make 12 from the remaining 5 and 7? Yes, 5+7. So (6-4)*(5+7) is the answer.",
        "answer": "(6-4)*(7+5)=24",
        "tree": {
            "node1": {"Problem": "4, 5, 6, 7", "parent": None, "Result": None},
            "node2": {"Problem": "(7*6), 4, 5", "parent": "node1", "Result": None},
            "node3": {"Problem": "(7*5), 4, 6", "parent": "node1", "Result": None},
            "node4": {"Problem": "(6*5), 4, 7", "parent": "node1", "Result": None},
            "node5": {"Problem": "(7-4), 5, 6", "parent": "node1", "Result": None},
            "node6": {"Problem": "((7-4)*6), 5", "parent": "node5", "Result": None},
            "node7": {"Problem": "((7-4)+6), 5", "parent": "node5", "Result": None},
            "node8": {"Problem": "(6-4), 5, 7", "parent": "node1", "Result": None},
            "node9": {"Problem": "(6-4), (7+5)", "parent": "node8", "Result": None},
            "node10": {"Problem": "((6-4)*(7+5))", "parent": "node9", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"}
        ]
    },
    {
        "numbers": [3, 6, 6, 7],
        "reasoning": "With two 6s, I'll start with 6*6=36. I have 3 and 7 left. I need to subtract 12. Can't make 12 from 3 and 7. Backtrack. What about 7*6=42? I have 3 and 6 left. I need to subtract 18. I can do that with 6*3=18. So, 7*6 - 6*3 = 24. This is a valid solution. But let's say I don't see that partition. I try 6+3=9, not 18. Backtrack. Okay, what about starting with 7*3=21? I need to add 3. I have two 6s. Can't make 3. Backtrack. What if I make 1? 6/6=1. Now I have 1, 3, 7. This is a good intermediate step. From here, let's try 7*3=21. Now I have 1 and 21. 1+21=22. Close. Backtrack to the 1,3,7 state. Instead of 7*3, let's try 7+1=8. Now I have 3 and 8. And 3*8=24. Got it.",
        "answer": "(6/6+7)*3=24",
        "tree": {
            "node1": {"Problem": "3, 6, 6, 7", "parent": None, "Result": None},
            "node2": {"Problem": "(6*6), 3, 7", "parent": "node1", "Result": None},
            "node3": {"Problem": "(7*6), 3, 6", "parent": "node1", "Result": None},
            "node4": {"Problem": "(7*3), 6, 6", "parent": "node1", "Result": None},
            "node5": {"Problem": "(6/6), 3, 7", "parent": "node1", "Result": None},
            "node6": {"Problem": "((6/6)*7), 3", "parent": "node5", "Result": None},
            "node7": {"Problem": "((6/6)+7), 3", "parent": "node5", "Result": None},
            "node8": {"Problem": "(((6/6)+7)*3)", "parent": "node7", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node5", "category": "verification"},
            {"from": "node5", "to": "node7", "category": "verification"},
            {"from": "node7", "to": "node8", "category": "verification"}
        ]
    },
    {
        "numbers": [2, 4, 6, 12],
        "reasoning": "My first instinct is that 12*2=24, but that leaves 4 and 6 unused, so it's invalid. Backtrack. Similarly, 6*4=24 is invalid because it leaves 2 and 12. Backtrack. Let's try a larger multiplication, 12*6=72. Now I have 2 and 4. 72/2=36, then 36-4=32. No. Backtrack. What about 12*4=48? I have 2 and 6. 48-6=42, then 42-2=40. Still no good. Backtrack. All the multiplications seem to fail or be invalid. This must be a simple addition or subtraction problem. Let's try adding them up. 12+6=18. Now I have 2, 4, and 18. Then 18+4=22. I have 2 and 22. And 22+2=24. It works.",
        "answer": "12+6+4+2=24",
        "tree": {
            "node1": {"Problem": "2, 4, 6, 12", "parent": None, "Result": None},
            "node2": {"Problem": "(12*2), 4, 6", "parent": "node1", "Result": None},
            "node3": {"Problem": "(6*4), 2, 12", "parent": "node1", "Result": None},
            "node4": {"Problem": "(12*6), 2, 4", "parent": "node1", "Result": None},
            "node5": {"Problem": "(12*4), 2, 6", "parent": "node1", "Result": None},
            "node6": {"Problem": "(12+6), 2, 4", "parent": "node1", "Result": None},
            "node7": {"Problem": "((12+6)+4), 2", "parent": "node6", "Result": None},
            "node8": {"Problem": "(((12+6)+4)+2)", "parent": "node7", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node6", "category": "verification"},
            {"from": "node6", "to": "node7", "category": "verification"},
            {"from": "node7", "to": "node8", "category": "verification"}
        ]
    },
    {
        "numbers": [3, 3, 5, 6],
        "reasoning": "Let's try the largest numbers first, 6*5=30. Now I have 3 and 3. I need to subtract 6. I can't get 6 from 3-3=0. Backtrack. What about 6*3=18? I have 3 and 5 left. 18+5=23, then 23+3=26. No. Backtrack. Okay, what if I start with the identical numbers? 3+3=6. This seems like a good intermediate state. I now have 5, 6, 6. Let's explore from here. Let's try 6*6=36. Now I have 5 and 36. 36-5=31. That's not it. Okay, let's backtrack to the 5,6,6 state. What if I try 6+6=12? Now I have 5 and 12. That won't work. This whole branch seems to be a dead end. Let me backtrack to the start and retry the 5,6,6 state. Instead of 6*6 or 6+6, I should try 6*5=30. Then I have 6 and 30. 30-6=24. Found it.",
        "answer": "(3+3)*5-6=24",
        "tree": {
            "node1": {"Problem": "3, 3, 5, 6", "parent": None, "Result": None},
            "node2": {"Problem": "(6*5), 3, 3", "parent": "node1", "Result": None},
            "node3": {"Problem": "(6*3), 3, 5", "parent": "node1", "Result": None},
            "node4": {"Problem": "((6*3)+5), 3", "parent": "node3", "Result": None},
            "node5": {"Problem": "(3+3), 5, 6", "parent": "node1", "Result": None},
            "node6": {"Problem": "((3+3)*6), 5", "parent": "node5", "Result": None},
            "node7": {"Problem": "((3+3)+6), 5", "parent": "node5", "Result": None},
            "node8": {"Problem": "((3+3)*5), 6", "parent": "node5", "Result": None},
            "node9": {"Problem": "(((3+3)*5)-6)", "parent": "node8", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node5", "category": "verification"},
            {"from": "node5", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"}
        ]
    },
    {
        "numbers": [2, 3, 7, 12],
        "reasoning": "Let's try multiplying the big numbers, 12 * 7 = 84. Too large. Backtrack. What about 12 * 3 = 36? I have 2 and 7 left. I need to subtract 12. I can't make 12 from 2 and 7. Backtrack. Let's try 12 * 2 = 24. That's invalid as it leaves 3 and 7 unused. Backtrack. Let's try subtraction first. 7-3=4. Now I have 2, 4, 12. This seems promising. Let me try multiplying from here: 12*4=48. Now I have 2 and 48. 48/2=24. This is a valid solution. But let's say I didn't see that. I tried 48-2=46. Not 24. Let's backtrack to the 2,4,12 state. From here, what if I try 12*2=24? That leaves 4 unused. Invalid. Backtrack to the 2,4,12 state again. Instead of multiplication, what if I do 12/2=6? Now I have 4 and 6. And 4*6=24. That's another solution!",
        "answer": "(7-3)*(12/2)=24",
        "tree": {
            "node1": {"Problem": "2, 3, 7, 12", "parent": None, "Result": None},
            "node2": {"Problem": "(12*7), 2, 3", "parent": "node1", "Result": None},
            "node3": {"Problem": "(12*3), 2, 7", "parent": "node1", "Result": None},
            "node4": {"Problem": "(12*2), 3, 7", "parent": "node1", "Result": None},
            "node5": {"Problem": "(7-3), 2, 12", "parent": "node1", "Result": None},
            "node6": {"Problem": "((7-3)*12), 2", "parent": "node5", "Result": None},
            "node7": {"Problem": "((7-3)*2), 12", "parent": "node5", "Result": None},
            "node8": {"Problem": "(7-3), (12/2)", "parent": "node5", "Result": None},
            "node9": {"Problem": "((7-3)*(12/2))", "parent": "node8", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node5", "category": "verification"},
            {"from": "node5", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"}
        ]
    },
    {
        "numbers": [2, 4, 4, 10],
        "reasoning": "My first thought is to try addition on all numbers, 10+4+4+2=20. Not 24. Backtrack. Let's try multiplying. 10*4=40. I have 2 and 4 left. 40-4=36, then 36-2=34. No. Backtrack. What about 4*4=16? I have 2 and 10 left. 16+10=26, then 26-2=24. This works. But let's say I missed it. Let's try 16+2=18, then 18+10=28. No. Backtrack. Okay, let's start with subtraction. 10-4=6. Now I have 2, 4, 6. Let's explore this. 6*4=24, but that leaves 2 unused. Invalid. Let me backtrack to the 2,4,6 state. From here, let's try 6*2=12. Then 12+4=16. Nope. Backtrack to the start. The solution must be a partition. 10*4=40. And the other pair is 4*4=16. Then 40-16=24. That works too.",
        "answer": "(10*4)-(4*4)=24",
        "tree": {
            "node1": {"Problem": "2, 4, 4, 10", "parent": None, "Result": None},
            "node2": {"Problem": "(10+4), 2, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "(10*4), 2, 4", "parent": "node1", "Result": None},
            "node4": {"Problem": "(4*4), 2, 10", "parent": "node1", "Result": None},
            "node5": {"Problem": "((4*4)+2), 10", "parent": "node4", "Result": None},
            "node6": {"Problem": "(10-4), 2, 4", "parent": "node1", "Result": None},
            "node7": {"Problem": "((10-4)*2), 4", "parent": "node6", "Result": None},
            "node8": {"Problem": "(10*4), (4*4)", "parent": "node1", "Result": None},
            "node9": {"Problem": "((10*4)-(4*4))", "parent": "node8", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 3, 7, 8],
        "reasoning": "Let's try multiplying the large numbers. 8 * 7 = 56. Too big. Backtrack. How about 8 * 3 = 24? That hits the target, but leaves 1 and 7 unused, so it's invalid. Backtrack. Let's try 7 * 3 = 21. I have 1 and 8 left. 21+8=29. No. Backtrack. Okay, let's start with subtraction. 8-1=7. Now I have 3, 7, 7. Let's explore this promising state. Try 7*7=49. Too big. Let's backtrack to the 3,7,7 state. How about 7+7=14? Now I have 3 and 14. 14+3=17. Still no. Backtrack to the start. My initial subtractions didn't work. Let's try 7-3=4. Now I have 1, 4, 8. Then 4-1=3. Now I have 3 and 8. And 3*8=24. That works!",
        "answer": "(7-3-1)*8=24",
        "tree": {
            "node1": {"Problem": "1, 3, 7, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(8*7), 1, 3", "parent": "node1", "Result": None},
            "node3": {"Problem": "(8*3), 1, 7", "parent": "node1", "Result": None},
            "node4": {"Problem": "(7*3), 1, 8", "parent": "node1", "Result": None},
            "node5": {"Problem": "(8-1), 3, 7", "parent": "node1", "Result": None},
            "node6": {"Problem": "((8-1)*7), 3", "parent": "node5", "Result": None},
            "node7": {"Problem": "((8-1)+7), 3", "parent": "node5", "Result": None},
            "node8": {"Problem": "(7-3), 1, 8", "parent": "node1", "Result": None},
            "node9": {"Problem": "((7-3)-1), 8", "parent": "node8", "Result": None},
            "node10": {"Problem": "(((7-3)-1)*8)", "parent": "node9", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 2, 6, 9],
        "reasoning": "Let's try multiplying the largest numbers first, 9 * 6 = 54. That's too big to be useful. Backtrack. How about 9 * 2 = 18? Now I have 1 and 6. 18+6=24. Oh, but that leaves the 1 unused, which is invalid. Backtrack. Let's try addition: 9+6=15. Then 15+2=17, and 17+1=18. Not a solution. Backtrack. Okay, I need a different approach. Let's start with subtraction. 9-1=8. This feels like a good intermediate step, creating a factor of 24. Now I have 2, 6, 8. From here, let's try 6*2=12. Now I have 8 and 12. 12+8=20. No. Let's backtrack to the 2,6,8 state. The 9-1=8 idea was good. What if I try 6/2=3? Now I have 8 and 3. And 8*3=24. That works!",
        "answer": "(9-1)*(6/2)=24",
        "tree": {
            "node1": {"Problem": "1, 2, 6, 9", "parent": None, "Result": None},
            "node2": {"Problem": "(9*6), 1, 2", "parent": "node1", "Result": None},
            "node3": {"Problem": "(9*2), 1, 6", "parent": "node1", "Result": None},
            "node4": {"Problem": "(9+6), 1, 2", "parent": "node1", "Result": None},
            "node5": {"Problem": "((9+6)+2), 1", "parent": "node4", "Result": None},
            "node6": {"Problem": "(9-1), 2, 6", "parent": "node1", "Result": None},
            "node7": {"Problem": "((9-1)-(6*2))", "parent": "node6", "Result": -4},
            "node8": {"Problem": "((9-1)*(6/2))", "parent": "node6", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node6", "category": "backtracking"},
            {"from": "node6", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node6", "category": "verification"},
            {"from": "node6", "to": "node8", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 4, 6, 7],
        "reasoning": "Let's try multiplying the largest numbers, 7 * 6 = 42. That's too big. Backtrack. What about 6 * 4 = 24? This is invalid as it leaves 1 and 7 unused. Backtrack. Okay, let's try addition. 7+6=13. Then 13+4=17, and 17+1=18. Not a solution. Backtrack. This suggests a more complex sequence of operations. Let's try making a factor of 24. 7-1=6. Now I have 4, 6, 6. Let's explore this intermediate state. Try 6*4=24, but this leaves the other 6 unused. Invalid. Let me backtrack to the 4,6,6 state. What if I try 6+6=12? Now I have 4 and 12. That won't make 24. Backtrack to the start. Let's try a different chain. 7+1=8. Now have 4, 6, 8. Then 8-4=4. Now have 4, 6. 4*6=24. That works!",
        "answer": "((7+1)-4)*6=24",
        "tree": {
            "node1": {"Problem": "1, 4, 6, 7", "parent": None, "Result": None},
            "node2": {"Problem": "(7*6), 1, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "(6*4), 1, 7", "parent": "node1", "Result": None},
            "node4": {"Problem": "(7+6), 1, 4", "parent": "node1", "Result": None},
            "node5": {"Problem": "(7-1), 4, 6", "parent": "node1", "Result": None},
            "node6": {"Problem": "((7-1)*4), 6", "parent": "node5", "Result": None},
            "node7": {"Problem": "(7+1), 4, 6", "parent": "node1", "Result": None},
            "node8": {"Problem": "((7+1)-4), 6", "parent": "node7", "Result": None},
            "node9": {"Problem": "(((7+1)-4)*6)", "parent": "node8", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node7", "category": "verification"},
            {"from": "node7", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"}
        ]
    },
    {
        "numbers": [2, 4, 6, 8],
        "reasoning": "My first thought is 8*3=24, but there's no 3. So let's try 6*4=24. This is invalid as it leaves 2 and 8 unused. Backtrack. What about 8*4=32? Now I have 2 and 6. I need to subtract 8. I can't make 8 from 2 and 6. Backtrack. Let's try 8*6=48. I need to divide by 2. I have 2 and 4. I could use the 2, but that leaves 4. Invalid. Backtrack. Okay, this seems to require a partition or a denominator. Let's try making a denominator. What if I make 2 from 4-2? Now I have the numbers 6, 8 and the expression (4-2). Let's explore. 8*6=48. And now I can do 48 / (4-2) which is 48/2=24. This works!",
        "answer": "(8*6)/(4-2)=24",
        "tree": {
            "node1": {"Problem": "2, 4, 6, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(6*4), 2, 8", "parent": "node1", "Result": None},
            "node3": {"Problem": "(8*4), 2, 6", "parent": "node1", "Result": None},
            "node4": {"Problem": "(8*6), 2, 4", "parent": "node1", "Result": None},
            "node5": {"Problem": "(8*6)/2, 4", "parent": "node4", "Result": None},
            "node6": {"Problem": "(8*6), (4-2)", "parent": "node4", "Result": None},
            "node7": {"Problem": "((8*6)/(4-2))", "parent": "node6", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node4", "category": "verification"},
            {"from": "node4", "to": "node6", "category": "verification"},
            {"from": "node6", "to": "node7", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 3, 6, 9],
        "reasoning": "Let's try a big multiplication first. 9 * 6 = 54. Too large. Backtrack. What about 9 * 3 = 27? I have 1 and 6 left. I need to subtract 3. I can't make 3 from 1 and 6. Backtrack. Okay, what about 6 * 3 = 18? I have 1 and 9. 18+9=27, then 27-1=26. No. Backtrack. Let's try division. 9/3=3. This is a good intermediate step. Now I have 1, 3, 6. Let's explore. 6*3=18. Now have 1 and 18. That makes 19. Not a solution. Okay, let's backtrack to the 1,3,6 state. The 9/3=3 start was good. Instead of multiplying, let's add 1. 3+1=4. Now I have 4 and 6. And 4*6=24. Got it.",
        "answer": "((9/3)+1)*6=24",
        "tree": {
            "node1": {"Problem": "1, 3, 6, 9", "parent": None, "Result": None},
            "node2": {"Problem": "(9*6), 1, 3", "parent": "node1", "Result": None},
            "node3": {"Problem": "(9*3), 1, 6", "parent": "node1", "Result": None},
            "node4": {"Problem": "(6*3), 1, 9", "parent": "node1", "Result": None},
            "node5": {"Problem": "(9/3), 1, 6", "parent": "node1", "Result": None},
            "node6": {"Problem": "((9/3)*6), 1", "parent": "node5", "Result": None},
            "node7": {"Problem": "((9/3)+1), 6", "parent": "node5", "Result": None},
            "node8": {"Problem": "(((9/3)+1)*6)", "parent": "node7", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node5", "category": "verification"},
            {"from": "node5", "to": "node7", "category": "verification"},
            {"from": "node7", "to": "node8", "category": "verification"}
        ]
    },
    {
        "numbers": [2, 3, 4, 6],
        "reasoning": "Let's try the obvious multiplications. 6*4=24. Invalid, leaves 2 and 3. Backtrack. 6*3=18. I have 2 and 4. 18+4=22, then 22+2=24. This works! But let's find another path for this example. Let's say from 18, I do 18+2=20, then 20+4=24. Same result. Okay, let's say I miss that. Backtrack. What about 4*3=12? I have 2 and 6. 12+6=18, 18+2=20. No. Backtrack. How about 4*2=8? I have 3 and 6. 8+6=14, 14+3=17. No. Backtrack. It seems a single starting multiplication isn't the only way. What about a full addition chain? 6+4=10. 10+3=13. 13+2=15. No. Let's go back to the 6*3=18 path that I found first. It seems to be the most direct.",
        "answer": "6*3+2+4=24",
        "tree": {
            "node1": {"Problem": "2, 3, 4, 6", "parent": None, "Result": None},
            "node2": {"Problem": "(6*4), 2, 3", "parent": "node1", "Result": None},
            "node3": {"Problem": "(4*3), 2, 6", "parent": "node1", "Result": None},
            "node4": {"Problem": "(4*2), 3, 6", "parent": "node1", "Result": None},
            "node5": {"Problem": "(6+4), 2, 3", "parent": "node1", "Result": None},
            "node6": {"Problem": "(6*3), 2, 4", "parent": "node1", "Result": None},
            "node7": {"Problem": "((6*3)+2), 4", "parent": "node6", "Result": None},
            "node8": {"Problem": "(((6*3)+2)+4)", "parent": "node7", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node6", "category": "verification"},
            {"from": "node6", "to": "node7", "category": "verification"},
            {"from": "node7", "to": "node8", "category": "verification"}
        ]
    },
    {
        "numbers": [3, 6, 7, 9],
        "reasoning": "Let's try multiplying the large numbers. 9 * 7 = 63. That's too big. Backtrack. How about 9 * 6 = 54? Still too big. Backtrack. Okay, 7 * 6 = 42. I need to subtract 18. I can't make 18 from 3 and 9. Backtrack. Let's try starting with division. 9/3=3. Now I have 3, 6, 7. This seems promising. Let me try multiplying from here, 7*6=42. Now I have 3 and 42. That won't work. Let me backtrack to the state with 3, 6, 7. The 9/3=3 start was good. Let's try 6*3=18. Now I have 7 and 18. 18+7=25. No. Backtrack to the start. Let's try subtraction. 9-6=3. Now I have 3, 3, 7. From here, 3*7=21. And now 21+3=24. That's the one.",
        "answer": "(9-6)*7+3=24",
        "tree": {
            "node1": {"Problem": "3, 6, 7, 9", "parent": None, "Result": None},
            "node2": {"Problem": "(9*7), 3, 6", "parent": "node1", "Result": None},
            "node3": {"Problem": "(9*6), 3, 7", "parent": "node1", "Result": None},
            "node4": {"Problem": "(7*6), 3, 9", "parent": "node1", "Result": None},
            "node5": {"Problem": "(9/3), 6, 7", "parent": "node1", "Result": None},
            "node6": {"Problem": "((9/3)*7), 6", "parent": "node5", "Result": None},
            "node7": {"Problem": "((9/3)*6), 7", "parent": "node5", "Result": None},
            "node8": {"Problem": "(9-6), 3, 7", "parent": "node1", "Result": None},
            "node9": {"Problem": "((9-6)*7), 3", "parent": "node8", "Result": None},
            "node10": {"Problem": "(((9-6)*7)+3)", "parent": "node9", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 2, 4, 12],
        "reasoning": "My first thought is 12*2=24, but that leaves 1 and 4 unused, which is invalid. Backtrack. What about 12*4=48? Too large. Backtrack. Let's try division. 12/2=6. Now I have 1, 4, 6. This is a good intermediate state to explore. From here, let's try 6*4=24. This leaves the 1 unused, so it's also invalid. I'll backtrack to the 1,4,6 state. From there, let's try 6*1=6. Now I have 4 and 6. 4+6=10. Not 24. This branch isn't working. I'll backtrack to the start. The solution must be a partition. I need two factors, like 8 and 3. Can I make 8? Yes, 12-4. Can I make 3 from the remaining 1 and 2? Yes, 1+2. So, (12-4)*(1+2) is the solution.",
        "answer": "(12-4)*(1+2)=24",
        "tree": {
            "node1": {"Problem": "1, 2, 4, 12", "parent": None, "Result": None},
            "node2": {"Problem": "(12*2), 1, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "(12*4), 1, 2", "parent": "node1", "Result": None},
            "node4": {"Problem": "(12/2), 1, 4", "parent": "node1", "Result": None},
            "node5": {"Problem": "((12/2)*4), 1", "parent": "node4", "Result": None},
            "node6": {"Problem": "((12/2)*1), 4", "parent": "node4", "Result": None},
            "node7": {"Problem": "(12-4), 1, 2", "parent": "node1", "Result": None},
            "node8": {"Problem": "(12-4), (1+2)", "parent": "node7", "Result": None},
            "node9": {"Problem": "((12-4)*(1+2))", "parent": "node8", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node4", "category": "backtracking"},
            {"from": "node4", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node7", "category": "verification"},
            {"from": "node7", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"}
        ]
    },
    {
        "numbers": [3, 4, 6, 6],
        "reasoning": "With two 6s, 6*6=36 seems like a good place to start. Now I have 3 and 4. I need to subtract 12. I can make 12 with 3*4. So (6*6)-(3*4)=24. That's a solution. But let's say I don't see that partition. From 36, I try 36-4=32, then 32-3=29. No good. Backtrack. Okay, what about 6*4=24? That leaves 3 and 6 unused. Invalid. Backtrack. How about 6*3=18? I have 4 and 6 left. 18+6=24, but that leaves 4 unused. Invalid. Backtrack. Let's try subtraction. 6-3=3. Now I have 3, 4, 6. Let's explore. 6*4=24, which leaves 3. Invalid. Backtrack to the 3,4,6 state. How about 6*3=18? Then 18+4=22. No. Let's go back to the first idea, the partition, which worked.",
        "answer": "(6*6)-(3*4)=24",
        "tree": {
            "node1": {"Problem": "3, 4, 6, 6", "parent": None, "Result": None},
            "node2": {"Problem": "(6*4), 3, 6", "parent": "node1", "Result": None},
            "node3": {"Problem": "(6*3), 4, 6", "parent": "node1", "Result": None},
            "node4": {"Problem": "(6-3), 4, 6", "parent": "node1", "Result": None},
            "node5": {"Problem": "((6-3)*6), 4", "parent": "node4", "Result": None},
            "node6": {"Problem": "(6*6), 3, 4", "parent": "node1", "Result": None},
            "node7": {"Problem": "((6*6)-4), 3", "parent": "node6", "Result": None},
            "node8": {"Problem": "(6*6), (3*4)", "parent": "node6", "Result": None},
            "node9": {"Problem": "((6*6)-(3*4))", "parent": "node8", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node6", "category": "backtracking"},
            {"from": "node6", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node6", "category": "verification"},
            {"from": "node6", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 4, 5, 8],
        "reasoning": "Let's try multiplying the largest numbers. 8 * 5 = 40. Too big. Backtrack. What about 8 * 4 = 32? I have 1 and 5. I need to subtract 8. I can't make 8 from 1 and 5. Backtrack. Okay, let's try 5 * 4 = 20. I have 1 and 8. 20+8=28, then 28-1=27. Not a solution. Backtrack. Let's try subtraction as a first step. 8-4=4. Now I have 1, 4, 5. This seems like a good intermediate state. Let's explore. 4*5=20. Now I have 1 and 20. 20+1=21. Not a solution. Let's backtrack to the 1,4,5 state. What if I try 4*1=4? Now I have 4 and 5. 4+5=9. Still no. Backtrack to the 1,4,5 state again. Let's try 5+1=6. Now I have 4 and 6. And 4*6=24. Got it.",
        "answer": "(8-4)*(5+1)=24",
        "tree": {
            "node1": {"Problem": "1, 4, 5, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(8*5), 1, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "(8*4), 1, 5", "parent": "node1", "Result": None},
            "node4": {"Problem": "(5*4), 1, 8", "parent": "node1", "Result": None},
            "node5": {"Problem": "(8-4), 1, 5", "parent": "node1", "Result": None},
            "node6": {"Problem": "((8-4)*5), 1", "parent": "node5", "Result": None},
            "node7": {"Problem": "((8-4)*1), 5", "parent": "node5", "Result": None},
            "node8": {"Problem": "((8-4)*(5+1))", "parent": "node5", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node5", "category": "verification"},
            {"from": "node5", "to": "node8", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 3, 7, 8],
        "reasoning": "Let's try multiplying the large numbers. 8 * 7 = 56. Too big. Backtrack. What about 8 * 3 = 24? This is invalid because it leaves 1 and 7 unused. Backtrack. Let's try 7 * 3 = 21. I have 1 and 8. 21+8-1=28. No. Backtrack. Okay, let's start with subtraction. 8-1=7. Now I have 3, 7, 7. This is a good intermediate state to explore. From here, let's try 7*7=49. Too big. Let me backtrack to the 3,7,7 state. What if I try 7+7=14? Now I have 3 and 14. That gives 17 or 11. No. Backtrack to the start. Let's try a different first move. 7-3=4. Now I have 1, 4, 8. Then 4-1=3. And now I have 3 and 8. And 3*8=24. Got it.",
        "answer": "((7-3)-1)*8=24",
        "tree": {
            "node1": {"Problem": "1, 3, 7, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(8*7), 1, 3", "parent": "node1", "Result": None},
            "node3": {"Problem": "(8*3), 1, 7", "parent": "node1", "Result": None},
            "node4": {"Problem": "(7*3), 1, 8", "parent": "node1", "Result": None},
            "node5": {"Problem": "(8-1), 3, 7", "parent": "node1", "Result": None},
            "node6": {"Problem": "((8-1)*7), 3", "parent": "node5", "Result": None},
            "node7": {"Problem": "((8-1)+7), 3", "parent": "node5", "Result": None},
            "node8": {"Problem": "(7-3), 1, 8", "parent": "node1", "Result": None},
            "node9": {"Problem": "((7-3)-1), 8", "parent": "node8", "Result": None},
            "node10": {"Problem": "(((7-3)-1)*8)", "parent": "node9", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node5", "category": "backtracking"},
            {"from": "node5", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"}
        ]
    },
    {
        "numbers": [3, 4, 5, 5],
        "reasoning": "With two 5s, let's start with 5*5=25. I have 3 and 4 left. I need to subtract 1. I can make 1 with 4-3. So, (5*5)-(4-3)=24. That's a solution. But let's say I didn't see that partition. From 25, I try 25-4=21, then 21-3=18. No. Backtrack. Let's try 5*4=20. I have 3 and 5 left. 20+5=25, then 25-3=22. No. Backtrack. What about 5*3=15? I have 4 and 5 left. 15+5=20, then 20+4=24. This is another solution! Let's say I miss this one too. I try 15+4=19, then 19+5=24. Okay, so many ways. Let's try a different first step. 5-3=2. Now I have 2, 4, 5. This is a good intermediate state. Let's explore. 5*4=20. Now I have 2 and 20. 20+2=22. Not 24. Backtrack to the 2,4,5 state. Let's try 5*2=10. Now I have 4 and 10. 10+4=14. No good. Backtrack to start. Let's stick with the first solution I found.",
        "answer": "(5*5)-(4-3)=24",
        "tree": {
            "node1": {"Problem": "3, 4, 5, 5", "parent": None, "Result": None},
            "node2": {"Problem": "(5*4), 3, 5", "parent": "node1", "Result": None},
            "node3": {"Problem": "((5*4)+5), 3", "parent": "node2", "Result": None},
            "node4": {"Problem": "(5*3), 4, 5", "parent": "node1", "Result": None},
            "node5": {"Problem": "((5*3)+5), 4", "parent": "node4", "Result": None},
            "node6": {"Problem": "(5-3), 4, 5", "parent": "node1", "Result": None},
            "node7": {"Problem": "((5-3)*5), 4", "parent": "node6", "Result": None},
            "node8": {"Problem": "(5*5), 3, 4", "parent": "node1", "Result": None},
            "node9": {"Problem": "(5*5), (4-3)", "parent": "node8", "Result": None},
            "node10": {"Problem": "((5*5)-(4-3))", "parent": "node9", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 4, 8, 9],
        "reasoning": "Let's try multiplying the large numbers. 9 * 8 = 72. That's too big. Backtrack. How about 9 * 4 = 36? I need to subtract 12. I can't make 12 from 1 and 8. Backtrack. Okay, what about 8 * 4 = 32? I need to subtract 8. I can make 8 from 9-1. So, 8*4 - (9-1) = 24. This is a valid solution. But let's say I miss the partition. I try 9+1=10, which gives 32-10=22. No. Backtrack. Okay, let's try addition. 9+8=17. Then 17+4=21, and 21+1=22. Not a solution. Backtrack. Let me go back to the idea of starting with subtraction. 9-1=8. Now I have 4, 8, 8. This is a good intermediate state. Let's multiply from here: 8*4=32. Now I have 8 and 32. And 32-8=24. Found it.",
        "answer": "(9-1)*4-8=24",
        "tree": {
            "node1": {"Problem": "1, 4, 8, 9", "parent": None, "Result": None},
            "node2": {"Problem": "(9*8), 1, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "(9*4), 1, 8", "parent": "node1", "Result": None},
            "node4": {"Problem": "(8*4), 1, 9", "parent": "node1", "Result": None},
            "node5": {"Problem": "((8*4)-(9+1))", "parent": "node4", "Result": 22},
            "node6": {"Problem": "(9+8), 1, 4", "parent": "node1", "Result": None},
            "node7": {"Problem": "(9-1), 4, 8", "parent": "node1", "Result": None},
            "node8": {"Problem": "((9-1)*4), 8", "parent": "node7", "Result": None},
            "node9": {"Problem": "(((9-1)*4)-8)", "parent": "node8", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node7", "category": "verification"},
            {"from": "node7", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 4, 7, 8],
        "reasoning": "Let's try multiplying the large numbers. 8 * 7 = 56. Too big. Backtrack. How about 8 * 4 = 32? I need to subtract 8. I can make 8 from 7+1. So, 8*4 - (7+1) = 24. This is a valid solution. But let's say I miss the partition and try 7-1=6. Then 32-6=26. No good. Backtrack. Okay, let's try 7 * 4 = 28. I need to subtract 4. I can't make 4 from 1 and 8. Backtrack. Let's start with subtraction. 8-1=7. Now I have 4, 7, 7. This seems like a promising intermediate step. From here, let's try 7*7=49. Too big. Let's backtrack to the 4,7,7 state. How about 7+7=14? Then 14-4=10. No. Let me backtrack to the start. Let's try addition first. 7+1=8. Now I have 4, 8, 8. Then 8*4=32. Now I have 8 and 32. And 32-8=24. Found another way.",
        "answer": "((7+1)*4)-8=24",
        "tree": {
            "node1": {"Problem": "1, 4, 7, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(8*7), 1, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "(8*4), 1, 7", "parent": "node1", "Result": None},
            "node4": {"Problem": "((8*4)-(7-1))", "parent": "node3", "Result": 26},
            "node5": {"Problem": "(7*4), 1, 8", "parent": "node1", "Result": None},
            "node6": {"Problem": "(8-1), 4, 7", "parent": "node1", "Result": None},
            "node7": {"Problem": "((8-1)*7), 4", "parent": "node6", "Result": None},
            "node8": {"Problem": "(7+1), 4, 8", "parent": "node1", "Result": None},
            "node9": {"Problem": "((7+1)*4), 8", "parent": "node8", "Result": None},
            "node10": {"Problem": "(((7+1)*4)-8)", "parent": "node9", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node10", "category": "calculation/deriviation"},
            {"from": "node10", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"},
            {"from": "node9", "to": "node10", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 5, 6, 7],
        "reasoning": "Let's try multiplying the large numbers. 7*6=42. Too big. Backtrack. What about 7*5=35? Need to subtract 11. I can't make 11 from 1 and 6. Backtrack. Okay, let's try 6*5=30. I need to subtract 6. I can make 6 from 7-1. So, 6*5-(7-1)=24. This is a solution. But let's say I miss the partition and try 7+1=8. Then 30-8=22. No good. Backtrack. Let's try starting with subtraction. 7-6=1. Now I have 1, 1, 5. Let's explore this. 1+1=2. Now I have 2 and 5. That's not useful. Let's backtrack to the 1,1,5 state. What if I do 5*1=5, then 5+1=6? No. This branch is a dead end. Backtrack to the start. Let's try another subtraction. 7-1=6. Now I have 5, 6, 6. Then 6*5=30. Now I have 6 and 30. And 30-6=24. Found it.",
        "answer": "(7-1)*5-6=24",
        "tree": {
            "node1": {"Problem": "1, 5, 6, 7", "parent": None, "Result": None},
            "node2": {"Problem": "(7*6), 1, 5", "parent": "node1", "Result": None},
            "node3": {"Problem": "(7*5), 1, 6", "parent": "node1", "Result": None},
            "node4": {"Problem": "(6*5), 1, 7", "parent": "node1", "Result": None},
            "node5": {"Problem": "((6*5)-(7+1))", "parent": "node4", "Result": 22},
            "node6": {"Problem": "(7-6), 1, 5", "parent": "node1", "Result": None},
            "node7": {"Problem": "(7-1), 5, 6", "parent": "node1", "Result": None},
            "node8": {"Problem": "((7-1)*5), 6", "parent": "node7", "Result": None},
            "node9": {"Problem": "(((7-1)*5)-6)", "parent": "node8", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node7", "category": "verification"},
            {"from": "node7", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"}
        ]
    },
    {
        "numbers": [2, 2, 3, 9],
        "reasoning": "Let's try multiplying the large numbers. 9 * 3 = 27. I have two 2s. I need to subtract 3. I can't make 3 from 2 and 2. Backtrack. What about 9 * 2 = 18? I have 2 and 3 left. 18+3=21, then 21+2=23. Not a solution. Backtrack. Okay, let's try addition. 9+3=12. I have two 2s. 12+2=14, 14+2=16. No. Backtrack. Let's try starting with subtraction. 9-3=6. Now I have 2, 2, 6. This is a good intermediate state. Let's explore. 6*2=12. Now I have 2 and 12. 12+2=14. Not 24. Let's backtrack to the 2,2,6 state. From here, let's try adding the 2s: 2+2=4. Now I have 4 and 6. And 4*6=24. Found it.",
        "answer": "(9-3)*(2+2)=24",
        "tree": {
            "node1": {"Problem": "2, 2, 3, 9", "parent": None, "Result": None},
            "node2": {"Problem": "(9*3), 2, 2", "parent": "node1", "Result": None},
            "node3": {"Problem": "(9*2), 2, 3", "parent": "node1", "Result": None},
            "node4": {"Problem": "((9*2)+3), 2", "parent": "node3", "Result": None},
            "node5": {"Problem": "(9+3), 2, 2", "parent": "node1", "Result": None},
            "node6": {"Problem": "(9-3), 2, 2", "parent": "node1", "Result": None},
            "node7": {"Problem": "((9-3)*2), 2", "parent": "node6", "Result": None},
            "node8": {"Problem": "(9-3), (2+2)", "parent": "node6", "Result": None},
            "node9": {"Problem": "((9-3)*(2+2))", "parent": "node8", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node5", "category": "calculation/deriviation"},
            {"from": "node5", "to": "node1", "category": "backtracking"},
            {"from": "node1", "to": "node6", "category": "calculation/deriviation"},
            {"from": "node6", "to": "node7", "category": "calculation/deriviation"},
            {"from": "node7", "to": "node6", "category": "backtracking"},
            {"from": "node6", "to": "node8", "category": "calculation/deriviation"},
            {"from": "node8", "to": "node9", "category": "calculation/deriviation"},
            {"from": "node9", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node6", "category": "verification"},
            {"from": "node6", "to": "node8", "category": "verification"},
            {"from": "node8", "to": "node9", "category": "verification"}
        ]
    },
    {
        "numbers": [2, 2, 3, 8],
        "reasoning": "I can make 4 by dividing 8 by 2. I can make 6 by multiplying 3 by 2. Then, I can multiply these two results, 4 and 6, to get 24.",
        "answer": "(8/2)*(3*2)=24",
        "tree": {
            "node1": {"Problem": "2, 2, 3, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(8/2), 2, 3", "parent": "node1", "Result": None},
            "node3": {"Problem": "(8/2), (3*2)", "parent": "node2", "Result": None},
            "node4": {"Problem": "((8/2)*(3*2))", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
    {
        "numbers": [2, 5, 6, 8],
        "reasoning": "I can make 30 by multiplying 5 by 6. I can make 6 by subtracting 2 from 8. Then, subtracting the second result from the first, 30 minus 6, gives 24.",
        "answer": "(5*6)-(8-2)=24",
        "tree": {
            "node1": {"Problem": "2, 5, 6, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(5*6), 2, 8", "parent": "node1", "Result": None},
            "node3": {"Problem": "(5*6), (8-2)", "parent": "node2", "Result": None},
            "node4": {"Problem": "((5*6)-(8-2))", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 2, 7, 8],
        "reasoning": "First, I'll multiply 8 by 2 to get 16. Then, I can add 7 to get 23. Finally, adding the last number, 1, gives me 24.",
        "answer": "8*2+7+1=24",
        "tree": {
            "node1": {"Problem": "1, 2, 7, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(8*2), 1, 7", "parent": "node1", "Result": None},
            "node3": {"Problem": "((8*2)+7), 1", "parent": "node2", "Result": None},
            "node4": {"Problem": "(((8*2)+7)+1)", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 5, 8, 9],
        "reasoning": "I can make 8 by subtracting 1 from 9. I can make 3 by subtracting 5 from 8. Multiplying these two results, 8 times 3, gives 24.",
        "answer": "(9-1)*(8-5)=24",
        "tree": {
            "node1": {"Problem": "1, 5, 8, 9", "parent": None, "Result": None},
            "node2": {"Problem": "(9-1), 5, 8", "parent": "node1", "Result": None},
            "node3": {"Problem": "(9-1), (8-5)", "parent": "node2", "Result": None},
            "node4": {"Problem": "((9-1)*(8-5))", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
    {
        "numbers": [4, 5, 6, 7],
        "reasoning": "I'll create two intermediate numbers. First, 6 minus 4 is 2. Second, 7 plus 5 is 12. Multiplying these two results, 2 times 12, gives 24.",
        "answer": "(6-4)*(7+5)=24",
        "tree": {
            "node1": {"Problem": "4, 5, 6, 7", "parent": None, "Result": None},
            "node2": {"Problem": "(6-4), 5, 7", "parent": "node1", "Result": None},
            "node3": {"Problem": "(6-4), (7+5)", "parent": "node2", "Result": None},
            "node4": {"Problem": "((6-4)*(7+5))", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 4, 7, 8],
        "reasoning": "First, I can add 7 and 1 to get 8. Then, I can multiply that result by 4 to get 32. Finally, subtracting the last number, 8, gives 24.",
        "answer": "(7+1)*4-8=24",
        "tree": {
            "node1": {"Problem": "1, 4, 7, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(7+1), 4, 8", "parent": "node1", "Result": None},
            "node3": {"Problem": "((7+1)*4), 8", "parent": "node2", "Result": None},
            "node4": {"Problem": "(((7+1)*4)-8)", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
    {
        "numbers": [3, 4, 6, 6],
        "reasoning": "I can make 36 by multiplying 6 by 6. I can make 12 by multiplying 3 by 4. Then, subtracting the second result from the first, 36 minus 12, gives 24.",
        "answer": "(6*6)-(3*4)=24",
        "tree": {
            "node1": {"Problem": "3, 4, 6, 6", "parent": None, "Result": None},
            "node2": {"Problem": "(6*6), 3, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "(6*6), (3*4)", "parent": "node2", "Result": None},
            "node4": {"Problem": "((6*6)-(3*4))", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 5, 6, 7],
        "reasoning": "First, I'll subtract 1 from 7 to get 6. Now I have 5, 6, and 6. I can multiply my result by 5 to get 30. Finally, subtracting the last 6 gives 24.",
        "answer": "(7-1)*5-6=24",
        "tree": {
            "node1": {"Problem": "1, 5, 6, 7", "parent": None, "Result": None},
            "node2": {"Problem": "(7-1), 5, 6", "parent": "node1", "Result": None},
            "node3": {"Problem": "((7-1)*5), 6", "parent": "node2", "Result": None},
            "node4": {"Problem": "(((7-1)*5)-6)", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
    {
        "numbers": [2, 3, 4, 6],
        "reasoning": "First, I'll multiply 6 by 3 to get 18. Then, I can add 4 to get 22. Finally, adding the last number, 2, gives me 24.",
        "answer": "6*3+4+2=24",
        "tree": {
            "node1": {"Problem": "2, 3, 4, 6", "parent": None, "Result": None},
            "node2": {"Problem": "(6*3), 2, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "((6*3)+4), 2", "parent": "node2", "Result": None},
            "node4": {"Problem": "(((6*3)+4)+2)", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 2, 4, 12],
        "reasoning": "I can make 8 by subtracting 4 from 12. I can make 3 by adding 1 and 2. Multiplying these two results, 8 times 3, gives 24.",
        "answer": "(12-4)*(1+2)=24",
        "tree": {
            "node1": {"Problem": "1, 2, 4, 12", "parent": None, "Result": None},
            "node2": {"Problem": "(12-4), 1, 2", "parent": "node1", "Result": None},
            "node3": {"Problem": "(12-4), (1+2)", "parent": "node2", "Result": None},
            "node4": {"Problem": "((12-4)*(1+2))", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 4, 8, 9],
        "reasoning": "First, I'll subtract 1 from 9 to get 8. Then, I can multiply that result by 4 to get 32. Finally, subtracting the last number, 8, gives 24.",
        "answer": "(9-1)*4-8=24",
        "tree": {
            "node1": {"Problem": "1, 4, 8, 9", "parent": None, "Result": None},
            "node2": {"Problem": "(9-1), 4, 8", "parent": "node1", "Result": None},
            "node3": {"Problem": "((9-1)*4), 8", "parent": "node2", "Result": None},
            "node4": {"Problem": "(((9-1)*4)-8)", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 4, 5, 8],
        "reasoning": "I can make 4 by subtracting 4 from 8. I can make 6 by adding 1 and 5. Multiplying these two results, 4 times 6, gives 24.",
        "answer": "(8-4)*(1+5)=24",
        "tree": {
            "node1": {"Problem": "1, 4, 5, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(8-4), 1, 5", "parent": "node1", "Result": None},
            "node3": {"Problem": "(8-4), (1+5)", "parent": "node2", "Result": None},
            "node4": {"Problem": "((8-4)*(1+5))", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 4, 5, 8],
        "reasoning": "I can make 4 by subtracting 4 from 8. I can make 6 by adding 1 and 5. Multiplying these two results, 4 times 6, gives 24.",
        "answer": "(8-4)*(1+5)=24",
        "tree": {
            "node1": {"Problem": "1, 4, 5, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(8-4), 1, 5", "parent": "node1", "Result": None},
            "node3": {"Problem": "(8-4), (1+5)", "parent": "node2", "Result": None},
            "node4": {"Problem": "((8-4)*(1+5))", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
    {
        "numbers": [2, 2, 6, 7],
        "reasoning": "First, I can add 7 and 2 to get 9. Then, I can multiply that result by the other 2 to get 18. Finally, adding the last number, 6, gives 24.",
        "answer": "(7+2)*2+6=24",
        "tree": {
            "node1": {"Problem": "2, 2, 6, 7", "parent": None, "Result": None},
            "node2": {"Problem": "(7+2), 2, 6", "parent": "node1", "Result": None},
            "node3": {"Problem": "((7+2)*2), 6", "parent": "node2", "Result": None},
            "node4": {"Problem": "(((7+2)*2)+6)", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
    {
        "numbers": [2, 4, 4, 10],
        "reasoning": "I can make 40 by multiplying 10 by 4. I can make 16 by multiplying 4 by 4. Then, subtracting the second result from the first, 40 minus 16, gives 24.",
        "answer": "(10*4)-(4*4)=24",
        "tree": {
            "node1": {"Problem": "2, 4, 4, 10", "parent": None, "Result": None},
            "node2": {"Problem": "(10*4), 2, 4, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "(10*4), (4*4)", "parent": "node2", "Result": None},
            "node4": {"Problem": "((10*4)-(4*4))", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
    {
        "numbers": [3, 4, 5, 5],
        "reasoning": "First, I'll multiply 5 by 5 to get 25. Then, I can subtract 4 to get 21. Finally, adding the last number, 3, gives me 24.",
        "answer": "5*5-4+3=24",
        "tree": {
            "node1": {"Problem": "3, 4, 5, 5", "parent": None, "Result": None},
            "node2": {"Problem": "(5*5), 3, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "((5*5)-4), 3", "parent": "node2", "Result": None},
            "node4": {"Problem": "(((5*5)-4)+3)", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
    {
        "numbers": [1, 2, 5, 8],
        "reasoning": "I can create the number 6 by subtracting 2 from 8. I can create the number 4 by subtracting 1 from 5. Multiplying these two results, 6 times 4, gives the final answer of 24.",
        "answer": "(8-2)*(5-1)=24",
        "tree": {
            "node1": {"Problem": "1, 2, 5, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(8-2), 1, 5", "parent": "node1", "Result": None},
            "node3": {"Problem": "(8-2), (5-1)", "parent": "node2", "Result": None},
            "node4": {"Problem": "((8-2)*(5-1))", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
        {
        "numbers": [2, 3, 5, 7],
        "reasoning": "First, I multiply 3 by 5 to get 15. Next, I add 7 to this result to get 22. Finally, adding the last number, 2, gives 24.",
        "answer": "3*5+7+2=24",
        "tree": {
            "node1": {"Problem": "2, 3, 5, 7", "parent": None, "Result": None},
            "node2": {"Problem": "(3*5), 2, 7", "parent": "node1", "Result": None},
            "node3": {"Problem": "((3*5)+7), 2", "parent": "node2", "Result": None},
            "node4": {"Problem": "(((3*5)+7)+2)", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
        {
        "numbers": [1, 5, 7, 9],
        "reasoning": "This can be solved in two parts. First, 9 minus 5 is 4. Second, 7 minus 1 is 6. Multiplying the results, 4 times 6, equals 24.",
        "answer": "(9-5)*(7-1)=24",
        "tree": {
            "node1": {"Problem": "1, 5, 7, 9", "parent": None, "Result": None},
            "node2": {"Problem": "(9-5), 1, 7", "parent": "node1", "Result": None},
            "node3": {"Problem": "(9-5), (7-1)", "parent": "node2", "Result": None},
            "node4": {"Problem": "((9-5)*(7-1))", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
        {
        "numbers": [1, 3, 6, 9],
        "reasoning": "First, I'll divide 9 by 3 to get 3. Then, I add 1 to that result to get 4. Finally, I multiply this by 6 to get the answer, 24.",
        "answer": "((9/3)+1)*6=24",
        "tree": {
            "node1": {"Problem": "1, 3, 6, 9", "parent": None, "Result": None},
            "node2": {"Problem": "(9/3), 1, 6", "parent": "node1", "Result": None},
            "node3": {"Problem": "((9/3)+1), 6", "parent": "node2", "Result": None},
            "node4": {"Problem": "(((9/3)+1)*6)", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
        {
        "numbers": [2, 4, 6, 8],
        "reasoning": "This solution has two main parts. First, I multiply 8 by 6 to get 48. For the second part, I subtract 2 from 4 to get 2. Finally, dividing the first result by the second, 48 divided by 2, gives 24.",
        "answer": "(8*6)/(4-2)=24",
        "tree": {
            "node1": {"Problem": "2, 4, 6, 8", "parent": None, "Result": None},
            "node2": {"Problem": "(8*6), 2, 4", "parent": "node1", "Result": None},
            "node3": {"Problem": "(8*6), (4-2)", "parent": "node2", "Result": None},
            "node4": {"Problem": "((8*6)/(4-2))", "parent": "node3", "Result": 24}
        },
        "walk": [
            {"from": "node1", "to": "node2", "category": "calculation/deriviation"},
            {"from": "node2", "to": "node3", "category": "calculation/deriviation"},
            {"from": "node3", "to": "node4", "category": "calculation/deriviation"},
            {"from": "node4", "to": "node1", "category": "verification"},
            {"from": "node1", "to": "node2", "category": "verification"},
            {"from": "node2", "to": "node3", "category": "verification"},
            {"from": "node3", "to": "node4", "category": "verification"}
        ]
    },
]


def get_item(numbers, reasoning, answer):
    prompt_template = f"""
    Now given a game 24 problem, we have 4 numbers: {numbers[0]}, {numbers[1]}, {numbers[2]}, and {numbers[3]}. 
    Your goal is to use all the 4 numbers and basic arithmetic operations (+ - * /) to obtain 24. 
    You must use each number exactly once, and you can use parentheses to change the order of operations.
    Please provide one feasible solution to this problem. 
    Your response should just be the answer containing only letter of the correct answer with no additional text—for example, 2*9+18/3=24
    """
    response = f"<think>{reasoning}</think>\n<answer>{answer}</answer>"
    
    item = deepcopy(template)
    item["prompt"][0]["content"] = prompt_template
    item["responses"][0] = response
    item["answers"][0] = answer
    item["reasonings"][0] = reasoning
    
    return item

syn_rewot_dir = get_result_dir(
    dataset_name="game24",
    model_name = "synthetic/synthetic",
    shot=0,
    template_type="reasoning_api_fake",
    response_length=404,
    num_samples=100,
    feature_noise=None,
    label_noise=0.0,
    train_step=0,
    data_mode="default",
    n_query=1,
    temperature=1.0,
    replicate_id=0,
)
os.makedirs(syn_rewot_dir, exist_ok=True)

syn_reasoning = []
for i, item in enumerate(synthetic_data):
    syn_reasoning.append(get_item(item["numbers"], item["reasoning"], item["answer"]))
    rewot = {
        "tree": item["tree"],
        "walk": item["walk"]
    }
    save_json(rewot, f"{syn_rewot_dir}/tree_vis_v3/{i}.json")

# pdb.set_trace()
syn_reasoning = pd.DataFrame(syn_reasoning)

syn_reasoning_dir = get_result_dir(
    dataset_name="game24",
    model_name = "synthetic/synthetic",
    shot=0,
    template_type="reasoning_api",
    response_length=404,
    num_samples=100,
    feature_noise=None,
    label_noise=0.0,
    train_step=0,
    data_mode="default",
    n_query=1,
    temperature=1.0,
    replicate_id=0,
)
os.makedirs(syn_reasoning_dir, exist_ok=True)
syn_reasoning.to_parquet(f"{syn_reasoning_dir}/test_default.parquet")
syn_reasoning.to_json(f"{syn_reasoning_dir}/test_default.json", orient='records', indent=2)





