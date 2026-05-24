import re

SYSTEM_MSG = """Your goal is to help users do various jobs by executing code.
You should:
1. Comprehend the user's requirements carefully & to the letter.
2. Describe what you plan to do in the code.
3. Provide Python code to run in a single code block, adding necessary libraries and custom functions before the code.
4. Add print statements for all key intermediate variables and for all conditions of all conditional statements in the code while ensuring correct syntax."""

REWRITE_MSG = """Please write this response again to make it better. You should:
1. Comprehend the user's requirements carefully & to the letter.
2. Describe what you plan to do in the code.
3. Provide Python code to run in a single code block, adding necessary libraries and custom functions before the code.
4. Add print statements for all key intermediate variables and for all conditions of all conditional statements in the code while ensuring correct syntax.
You should preserve everything in the original code, even if it contains a syntactical error or a bug that prevents the code from running.
Return the response only. Do not say anything else. Act like writing the response for the first time."""

COMPLETE_MSG = """Here is a partially completed code as response:
{code}
You should write the complete response as follows:
1. Comprehend the user's requirements carefully & to the letter.
2. Describe what you plan to do in the code.
3. Provide the complete Python code to run in a single code block, adding necessary libraries and custom functions before the code.
4. Identify all key intermediate variables in the code and add print statements for them.
You should start the code with the originally provided code and preserve everything in it, even if it contains a syntactical error or a bug that prevents the code from running.
Return the response only. Do not say anything else. Act like writing the response for the first time.
"""

COMPLETE_MSG_WITH_EXAMPLE = """Here is a partially completed code as response:
{code}
You should write the complete response as follows:
1. Comprehend the user's requirements carefully & to the letter.
2. Describe what you plan to do in the code.
3. Provide the complete Python code to run in a single code block, adding necessary libraries and custom functions before the code.
4. Identify all key intermediate variables in the code and add print statements for them.
You should start the code with the originally provided code and preserve everything in it, even if it contains a syntactical error or a bug that prevents the code from running.
For example, you can add the following code after the provided code:
```python
{code_example}
```
By executing this completion with tests, you get the following feedback:
{feedback_example}
Now, you should write your own response. Write all code in a single code block.
Return the response only. Do not say anything else. Act like writing the response for the first time.
"""

WRITE_CALLS_MSG = """Write up to five calling examples to execute the code in the response.
Write the calling examples in the way that they will be directly attached to the end of the code when running.
Cover as many input scenarios as possible, including edge cases and complicated inputs.
Do not write invalid inputs that are guaranteed not to appear.
Do not predict the expected outputs or return values.
Return the calling examples only. Do not include the original code."""

CRITIC_MSG_CORRECTNESS = """You are an expert at evaluating the quality of code.
As an impartial evaluator, please assess the correctness of a code generation assistant’s response to a user’s problem using the 5-point scoring system described below. The code will go through debugging with print statements showing the running process. Please grade the code based on the satisfaction of each criterion:
1: It means the code is relevant and provides some information related to the problem, even if it is incomplete, can not be run, or contains irrelevant content.
2: It means the code can sometimes run and produce outputs, but contains bugs or logical errors so that it does not make the correct outputs to the problem.
3: It means the code can solve the simple cases of the problem, but failes in difficult, complicated, or edge cases, misses necessary error handling, or contains any security vulnerabilities.
4: It means the code solves the problem directly and comprehensively, having correct logic and covering every possible input case. Cases guaranteed not to appear do not need to be covered.
5: It means the code is optimized for time and space complexity, reflects the best practices specific to the algorithm, language, or framework used, and perfectly meets the user's specific requirements.

#Problem Begins#
{problem}
#Problem Ends#

The response contains the following code:
{code}

By executing the code, you get the following output:
#Output Begins#
{exec_result}
#Output Ends#

After examining the user’s instruction and the response:
1. Provide an analysis of the response, carefully monitoring the execution of the code for any errors or exceptions that may arise and paying close attention to the output produced by the execution. Use the format: "Analysis: <analysis>"
2. Justify your score by checking whether the code meets each criterion using the format: "Justify: <check criteria>".
3. Conclude with the score using the format: "Score: <total points>"
"""

def is_valid_critic(feedback):
    scores = re.findall(r'Score:\D*?(\d+(?:\.\d+)?)', feedback)
    if len(scores) > 0:
        for score in scores[::-1]:
            if float(score) >= 0 and float(score) <= 5:
                return True
    return False
