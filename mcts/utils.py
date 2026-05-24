import re

code_md_template = f"```python" + '\n{code}\n```'

def contain_code(text, language):
    return len(re.findall(r'```' + language + r'\n([\d\D]*?)\n```', text)) > 0

def get_code_str(text):
    blocks = re.findall(r'```(.*?)\n([\d\D]*?)\n```', text)
    langs = set()
    langs = [x for x in [b[0] for b in blocks] if not (x in langs or langs.add(x))]
    code = {}
    for l in langs:
        code[l] = '\n'.join([b[1].rstrip() for b in blocks if b[0] == l])
    return code

def get_code_blocks(text):
    blocks = re.findall(r'```(.*?)\n([\d\D]*?)\n```', text)
    langs = set()
    langs = [x for x in [b[0] for b in blocks] if not (x in langs or langs.add(x))]
    code = {}
    for l in langs:
        code[l] = [b[1].rstrip() for b in blocks if b[0] == l]
    return code

def remove_code_comments(code):
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'(\'\'\'(.*?)\'\'\'|"""(.*?)""")', '', code, flags=re.DOTALL)
    code = "\n".join(line.rstrip() for line in code.splitlines() if line.strip())
    return code
