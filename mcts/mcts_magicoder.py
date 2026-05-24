"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from collections import defaultdict, namedtuple
import re
import ast
import math
import random
import tokenize
from io import StringIO
from typing import Tuple, List, Set
from astroid import nodes
from astroid.builder import AstroidBuilder
from execution import syntax_correct_python, run_python_with_server
from prompts_mcts import SYSTEM_MSG, REWRITE_MSG, COMPLETE_MSG, COMPLETE_MSG_WITH_EXAMPLE, CRITIC_MSG_CORRECTNESS, WRITE_CALLS_MSG, is_valid_critic
from utils import contain_code, get_code_str, get_code_blocks, remove_code_comments, code_md_template

def get_function(code, entry=None) -> Tuple[str, str, str]:
    """ function part, part after the function, entry name """
    tree = AstroidBuilder().string_build(code)
    cutoff, function_name = None, entry
    for ele in tree.body:
        if isinstance(ele, nodes.FunctionDef) or isinstance(ele, nodes.ClassDef):
            cutoff = ele.end_lineno
            if entry is None:
                function_name = ele.name
            if entry is not None and ele.name == entry:
                break
    return ('\n'.join(code.split('\n')[:cutoff]).strip(), '\n'.join(code.split('\n')[cutoff:]).strip(), function_name) if cutoff is not None else (None, None, None)

def is_printing_variables(print_code):
    if not print_code.lstrip().startswith('print'):
        return False
    tree = ast.parse(print_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            for arg in node.args:
                for sub_node in ast.walk(arg):
                    if isinstance(sub_node, ast.Load):
                        return True
            return False
    return False

def get_test(instruction, response, model):
    conversation = [
        {'role': 'system', 'content': SYSTEM_MSG},
        {'role': 'user', 'content': instruction},
        {'role': 'assistant', 'content': response},
        {'role': 'user', 'content': WRITE_CALLS_MSG}
    ]

    for retry in range(3):
        cases = model.invoke(conversation)
        if contain_code(cases, 'python'):
            return get_code_str(cases)['python'].strip()
        else:
            print("No Python code!")
    print("No test!")
    return ''

def get_segments(code):
    # segment with logical lines
    endline_places = [t.end for t in tokenize.generate_tokens(StringIO(code).readline) if t.type == tokenize.NEWLINE]
    lines = code.split('\n')
    split_places, current_line, current_index = [0], 0, 0
    for token in endline_places:
        while current_line < token[0] - 1:
            current_index += len(lines[current_line]) + 1
            current_line += 1
        split_places.append(current_index + token[1])
    logical_lines = [code[ split_places[i]: split_places[i + 1] ] for i in range(len(split_places) - 1)]
    logical_lines = [l[:-1] if l.endswith('\n') else l for l in logical_lines]
    # segment with print
    blocks, block_lines = [], []
    for l in logical_lines:
        block_lines.append(l)
        if is_printing_variables(l.lstrip()):
            blocks.append(block_lines)
            block_lines = []
    if len(block_lines) > 0:
        blocks.append(block_lines)
    blocks_new = []
    for b in blocks:
        if len(blocks_new) > 0 and b[0].lstrip().startswith('print'):
            blocks_new[-1].extend(b)
        else:
            blocks_new.append(b)
    # segment with indent
    blocks_new_2 = []
    for b in blocks_new:
        block_lines = []
        previous_indent = 0
        for l in b:
            if len(l.strip()) == 0:
                block_lines.append(l)
            else:
                indent = len(l) - len(l.lstrip(' '))
                if indent < previous_indent:
                    blocks_new_2.append('\n'.join(block_lines))
                    block_lines = []
                block_lines.append(l)
                previous_indent = indent
        if len(block_lines) > 0:
            blocks_new_2.append('\n'.join(block_lines))
    return blocks_new_2

def extract_code(response, entry):
    """ func, code_suffix, entry """
    if contain_code(response, 'python'):
        code = get_code_str(response)['python'].strip()
        if syntax_correct_python(code):
            func, suffix, entry = get_function(code, entry=entry)
            if func:
                return func, suffix, entry
            else:
                return code, '', entry
        else:
            print("Syntax error!")
    return None, None, None

def get_response(instruction, model, base_response=None, entry=None):
    """ complete response, code, suffix code, entry """
    conversation = [
        {'role': 'system', 'content': SYSTEM_MSG},
        {'role': 'user', 'content': instruction},
    ]
    if base_response:
        conversation.extend([
            {'role': 'assistant', 'content': base_response},
            {'role': 'user', 'content': REWRITE_MSG},
        ])
    while True:
        response = model.invoke(conversation)
        print(response)
        func, suffix, entry = extract_code(response, entry)
        if func is None:
            if random.random() < 0.5:
                conversation = conversation[:2]
            else:
                conversation = conversation[:2] + [
                    {'role': 'assistant', 'content': base_response if base_response else response},
                    {'role': 'user', 'content': REWRITE_MSG},
                ]
            print("Rewrite response!")
        else:
            return response, func, suffix, entry
                
def get_continual_response(instruction, previous_code, model, entry=None, code_example=None, feedback_example=None):
    if code_example is not None and feedback_example is not None:
        conversation = [
            {'role': 'system', 'content': SYSTEM_MSG},
            {'role': 'user', 'content': instruction.rstrip() + '\n' + COMPLETE_MSG_WITH_EXAMPLE.format(code=previous_code, code_example=code_example, feedback_example=feedback_example)},
        ]
    else:
        conversation = [
            {'role': 'system', 'content': SYSTEM_MSG},
            {'role': 'user', 'content': instruction.rstrip() + '\n' + COMPLETE_MSG.format(code=previous_code)},
        ]
    for retry in range(3):
        response = model.invoke(conversation)
        if contain_code(response, 'python'):
            code_blocks = get_code_blocks(response)['python']
            if len(code_blocks) == 1:
                code = get_code_str(response)['python'].strip()
                if syntax_correct_python(code):
                    func, suffix, entry = get_function(code, entry=entry)
                    if func:
                        if func.startswith(previous_code) and func != previous_code and func[len(previous_code):].lstrip(' ').startswith('\n'):
                            return response, func, entry
                        else:
                            conversation = conversation[:2] + [
                                {'role': 'assistant', 'content': response},
                                {'role': 'user', 'content': "Your code must start with the following:\n```python\n" + code_md_template.format(code=previous_code) + '\n\n```\nNow, rewrite the response. Return the rewritten response only. Do not say anything else.'},
                            ]
    return None, None, None

CodeStep = namedtuple("CodeStep", "depth segment terminal response")

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self,
                 model,
                 branching=3,
                 exploration_weight=math.sqrt(2),
                 tests=None,
                 use_reflection=False):
        # MCTS related
        self.model = model
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.nodes: List[CodeStep] = []
        self.children = defaultdict(set)  # children of each node, key: index
        self.parent = dict() # parent of each node, eky: index
        self.branching = branching # number of minimum branches for an expanded node ("minimum" comes from the condition in `_expand`)
        self.remain_branches = defaultdict(lambda: self.branching) # key: index
        self.exploration_weight = exploration_weight
        self.root = None
        self.initial_unexplored_nodes = []
        self.leaves: Set[int] = set() # indices
        self.critic = defaultdict(list) # leaf_index: List[( tests_and_outputs: [(tests: str, outputs: str)], feedback: Optional[str], reward: float )]
        self.use_tests_for_simulation = tests is not None
        self.tests = tests if self.use_tests_for_simulation else []
        self.use_reflection = use_reflection
        self.leaf_used_as_critic = defaultdict(lambda: False) # index: bool
        assert isinstance(self.tests, List)

        # Task related
        self.instruction = None
        self.entry = None

    def print(self, node, level=0):
        print(" " * (level * 4) + str(node.terminal) + ' ' + str([node.segment]))
        for child in self.children[node]:
            self.print(child, level + 1)

    def initiate(self, instruction, response=None, rewrite=False, entry=None, lock=None):
        "Initiate a tree with a program"
        self.instruction = instruction
        if not response:
            rewrite = True
        func = None
        if not rewrite:
            func, suffix, entry = extract_code(response, entry)
        if not func or rewrite:
            response, func, code_suffix, entry = get_response(instruction, self.model, base_response=response, entry=entry)
        print(f"Full response:\n{response}")
        if not self.tests:
            test = get_test(instruction, response, self.model)
            print(f"Tests:\n{test}")
            self.tests.append(test)
        segments = get_segments(func)
        print(f"Number of segments: {len(segments)}\n" + '\n======\n'.join(segments))
        self.root = CodeStep(depth=0, segment='', terminal=False, response='')
        self.nodes.append(self.root)
        node_idx = len(self.nodes) - 1
        for i, s in enumerate(segments):
            child = CodeStep(depth=i + 1, segment=s, terminal=(i == len(segments) - 1), response=response if (i == len(segments) - 1) else '')
            child_idx = len(self.nodes)
            if i != len(segments) - 1:
                self.initial_unexplored_nodes.append(child_idx)
            self.nodes.append(child)
            self.children[node_idx].add(child_idx)
            self.parent[child_idx] = node_idx
            self.remain_branches[node_idx] -= 1
            node_idx = child_idx
        self.leaves.add(node_idx)
        reward, tests_and_outputs, feedback = self._simulate(node_idx, lock)
        self.critic[node_idx].append((tests_and_outputs, feedback, reward))
        self._backpropagate(node_idx, reward)
        return self.root

    def do_rollout(self, node: CodeStep, lock=None):
        "Make the tree one layer better."
        node = self.initial_unexplored_nodes.pop(0) if self.initial_unexplored_nodes else self._select()
        path = [0]
        while node in self.parent:
            path.insert(1, node)
            node = self.parent[node]
        node = path[-1]
        print(f"Selected path:\n{path}")
        leaf = self._expand(node)
        if leaf is None:
            leaf = random.choice(list(self.leaves))
        else:
            self.leaves.add(leaf)
        reward, tests_and_outputs, feedback = self._simulate(leaf, lock)
        self.critic[leaf].append((tests_and_outputs, feedback, reward))
        self._backpropagate(leaf, reward)

    def dump(self):
        # """
        # paths: List[
        #     Tuple[
        #         path: List[ Tuple[ segment, q, n ] ],
        #         full response
        #     ]
        # ]
        # """
        """
        paths: List[
            Tuple[
                path: List[ Tuple[ segment, q, n ] ],
                full response: str,
                critic: List[ Tuple[ List[ tests: str, outputs: str ], feedback: Optional[str] ] ],
            ]
        ]
        """
        paths = []
        for leaf in self.leaves:
            path = []
            ptr = leaf
            while True:
                path.append((self.nodes[ptr].segment, self.Q[ptr], self.N[ptr]))
                if ptr not in self.parent:
                    break
                ptr = self.parent[ptr]
            paths.append((path[::-1], self.nodes[leaf].response, self.critic[leaf]))
        print(len(self.leaves), len(paths))
        return paths
    
    def _get_leaves(self, node_idx: int):
        return [i for j in [self._get_leaves(c) for c in self.children[node_idx]] for i in j] if node_idx in self.children else [node_idx]

    def _select(self):
        "Find an unexplored descendent of root"
        node_idx = 0
        while True:
            if self.nodes[node_idx].terminal or self.remain_branches[node_idx] > 0:
                return node_idx
            node_idx = self._uct_select(node_idx)

    def _expand(self, node_idx: int):
        "Update the `children` dict with the children of `node`"
        if self.nodes[node_idx].terminal:
            return node_idx
        response, func, segments = None, None, []
        if self.nodes[node_idx].depth == 0:
            # Write from scratch
            for retry in range(3):
                response, func, code_suffix, entry = get_response(self.instruction, self.model, entry=self.entry)
                if func is not None:
                    break
            if func is None:
                return None
            segments = get_segments(func)
        else:
            # Continual writing
            previous_code = self._gather_code_path(node_idx)
            if self.use_reflection:
                # Get previous complete code as feedback
                leaves = self._get_leaves(node_idx)
                all_leaves_used = False
                leaves_available = [l for l in leaves if not self.leaf_used_as_critic[l]]
                if len(leaves_available) == 0:
                    leaves_available = leaves
                    all_leaves_used = True
                random.shuffle(leaves_available)
                leaf_values = [self.Q[l] / self.N[l] for l in leaves_available]
                # TODO: best or worst or random choice?
                leaf_chosen = leaves_available[ leaf_values.index(max(leaf_values) if all_leaves_used else min(leaf_values)) ]
                self.leaf_used_as_critic[leaf_chosen] = True
                # TODO: best or worst or random choice?
                # resort to worst for now
                tests_and_outputs, feedback, reward = min(self.critic[leaf_chosen], key=lambda x: x[2])
                # critic = random.choice(self.critic[leaf_chosen])
                code_example = self._gather_code_path(leaf_chosen)[len(previous_code):]
                code_example = '\n'.join([line for line in code_example.split('\n') if line.strip()])
                feedback_example = ""
                if feedback is None:
                    # public tests
                    feedback_example = "\n".join([f"Test:\n```python\n{test}\n```\nOutput:\n```\n{output}\n```" for test, output in tests_and_outputs])
                else:
                    # model judge
                    test, output = tests_and_outputs[0]
                    feedback_example = f"Test:\n```python\n{test}\n```\nOutput:\n```\n{output}\n```\nFeedback:\n{feedback}"
            else:
                code_example = None
                feedback_example = None
            for retry in range(3):
                response, func, entry = get_continual_response(self.instruction, previous_code=previous_code, model=self.model, code_example=code_example, feedback_example=feedback_example)
                if func is not None:
                    break
            if func is None:
                return None
            segments_full = get_segments(func)
            for i, s in enumerate(segments_full):
                if previous_code.startswith(s):
                    previous_code = previous_code[len(s):]
                    if len(previous_code) > 0:
                        assert previous_code.startswith('\n'), (s, previous_code)
                        previous_code = previous_code[1:]
                else:
                    assert s.startswith(previous_code), (s, previous_code)
                    segments = [ s[len(previous_code):] ] + segments_full[i + 1:]
                    break
        
        for i, s in enumerate(segments):
            # if duplicate segment
            child_idx = None
            for c in self.children.get(node_idx, set()):
                if remove_code_comments(s) == remove_code_comments(self.nodes[c].segment):
                    # duplicate segment, follow
                    child_idx = c
                    break
            if child_idx is None:
                # if no duplicate segment 
                child = CodeStep(depth=i + 1, segment=s, terminal=(i == len(segments) - 1), response=response if (i == len(segments) - 1) else '')
                child_idx = len(self.nodes)
                self.nodes.append(child)
                self.children[node_idx].add(child_idx)
                self.parent[child_idx] = node_idx
            self.remain_branches[node_idx] -= 1
            node_idx = child_idx
        return node_idx
    
    def _gather_code_path(self, node_idx: int):
        code = []
        while node_idx in self.parent:
            code.append(self.nodes[node_idx].segment)
            node_idx = self.parent[node_idx]
        return '\n'.join(code[::-1]).rstrip()

    def _simulate(self, node_idx: int, lock = None):
        "score, [(tests, outputs)], feedback(optional)"
        code = self._gather_code_path(node_idx)
        len_threshold = 5000
        test = random.choice(self.tests)
        code += '\n' + test
        print(f"Gathered code:\n{code}")
        exec_out, exec_err, log = run_python_with_server(code, lock=lock)
        # Lengthy output
        if len(log) > len_threshold:
            log = log[:len_threshold // 2] + ' ... ' + log[-len_threshold // 2:]
        print(f"Output:\n{log}")
        model_input = CRITIC_MSG_CORRECTNESS.format(problem=self.instruction, code=code_md_template.format(code=code), exec_result=log)
        while True:
            feedback = self.model.generate(model_input)
            print(f"Feedback: {feedback}")
            if is_valid_critic(feedback):
                return float(re.findall(r'Score:\D*?(\d+(?:\.\d+)?)', feedback)[-1]) / 5, [(test, log)], feedback

    def _backpropagate(self, node_idx: int, reward):
        "Send the reward back up to the ancestors of the leaf"
        while node_idx is not None:
            self.N[node_idx] += 1
            self.Q[node_idx] += reward
            node_idx = self.parent.get(node_idx, None)

    def _uct_select(self, node_idx):
        "Select a child of node, balancing exploration & exploitation"

        log_N_vertex = math.log(self.N[node_idx])
        return max(self.children[node_idx], key=lambda n: self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(log_N_vertex / self.N[n]))
