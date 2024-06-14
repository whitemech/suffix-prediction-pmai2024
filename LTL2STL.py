import random


def infix_to_prefix(formula):
    precedence = {'!': 3, 'X': 2, 'F': 2, 'G': 2, '&': 1, '|': 1, 'i': 0, 'e': 0}
    stack = []
    prefix_formula = []

    def is_operator(token):
        return token in precedence.keys()

    def has_higher_precedence(op1, op2):
        return precedence[op1] > precedence[op2]

    def split_formula(formula):
        return formula.replace('(', ' ( ').replace(')', ' ) ').split()

    tokens = split_formula(formula)

    for token in reversed(tokens):
        if token == ')':
            stack.append(token)
        elif token == '(':
            while stack and stack[-1] != ')':
                prefix_formula.append(stack.pop())
            stack.pop()
        elif is_operator(token):
            while (stack and is_operator(stack[-1]) and
                   has_higher_precedence(stack[-1], token)):
                prefix_formula.append(stack.pop())
            stack.append(token)
        else:
            prefix_formula.append(token)

    while stack:
        prefix_formula.append(stack.pop())

    return ' '.join(reversed(prefix_formula))

def prefix_LTL_to_STL(formula):
        if formula.startswith('!'):
            ##print("negation formula:", formula)
            arg, rest = prefix_LTL_to_STL(formula[2:])
            ##print(f"neg: arg:{arg}, rest:{rest}")
            return f"('neg',({arg}))", rest
        elif formula.startswith('X'):
            #print("next")
            arg, rest = prefix_LTL_to_STL(formula[2:])
            #print(f"next: arg:{arg}, rest:{rest}")
            #TODO: check with (1,1)
            return f"('evntually', (1,2), ({arg}))", rest
        elif formula.startswith('F'):
            #print("eventually")
            arg, rest = prefix_LTL_to_STL(formula[2:])
            #print(f"event: arg:{arg}, rest:{rest}")
            return f"('eventually', (0, timeunits -1), ({arg}))", rest
        elif formula.startswith('G'):
            #print("globally")
            arg, rest = prefix_LTL_to_STL(formula[2:])
            #print(f"glob: arg:{arg}, rest:{rest}")
            return f"('always', (0, timeunites-1), ({arg}))", rest
        elif formula.startswith('&'):
            #print("conjunction")
            arg1, rest = prefix_LTL_to_STL(formula[2:])
            arg2, rest = prefix_LTL_to_STL(rest)
            #print(f"conjun: arg1:{arg1}, arg2:{arg2}, rest:{rest})")
            return f"('and',({arg1}), ({arg2}))", rest
        elif formula.startswith('|'):
            #print("disjunction")
            arg1, rest = prefix_LTL_to_STL(formula[2:])
            arg2, rest = prefix_LTL_to_STL(rest)
            #print(f"disjun: arg1:{arg1}, arg2:{arg2}, rest:{rest})")
            return f"('or', ({arg1}), ({arg2}))", rest
        elif formula.startswith('i'):
            ##print("implication")
            arg1, rest = prefix_LTL_to_STL(formula[2:])
            arg2, rest = prefix_LTL_to_STL(rest)
            #print(f"implic: arg1:{arg1}, arg2:{arg2}, rest:{rest})")
            return f"('implication', ({arg1}), ({arg2}))", rest
        elif formula.startswith('e'):
            #print("equivalence")
            arg1, rest = prefix_LTL_to_STL(formula[2:])
            arg2, rest = prefix_LTL_to_STL(rest)
            #print(f"equiv: arg1:{arg1}, arg2:{arg2}, rest:{rest})")
            return f"('equivalence', ({arg1}), ({arg2}))", rest
        elif formula.startswith('U'):
            #print("until")
            arg1, rest = prefix_LTL_to_STL(formula[2:])
            arg2, rest = prefix_LTL_to_STL(rest)
            #print(f"until: arg1:{arg1}, arg2:{arg2}, rest:{rest})")
            return f"('until', (0, timeunites-1), ({arg1}), ({arg2}))", rest
        elif formula.startswith('c'):
            #print("symbol")
            if len(formula) > 2:
                #print("symbol rest: ", formula[3:])
                return formula[:2], formula[3:]
            else:
                #print("symbol rest: ")
                return formula[:2], ""


def prefix_LTL_to_scarlet(formula):
    if formula.startswith('!'):
        ##print("negation formula:", formula)
        arg, rest = prefix_LTL_to_scarlet(formula[2:])
        ##print(f"neg: arg:{arg}, rest:{rest}")
        return f"!({arg})", rest
    elif formula.startswith('X'):
        # print("next")
        arg, rest = prefix_LTL_to_scarlet(formula[2:])
        # print(f"next: arg:{arg}, rest:{rest}")
        return f"X({arg})", rest
    elif formula.startswith('F'):
        # print("eventually")
        arg, rest = prefix_LTL_to_scarlet(formula[2:])
        # print(f"event: arg:{arg}, rest:{rest}")
        return f"F({arg})", rest
    elif formula.startswith('G'):
        # print("globally")
        arg, rest = prefix_LTL_to_scarlet(formula[2:])
        # print(f"glob: arg:{arg}, rest:{rest}")
        return f"G({arg})", rest
    elif formula.startswith('&'):
        # print("conjunction")
        arg1, rest = prefix_LTL_to_scarlet(formula[2:])
        arg2, rest = prefix_LTL_to_scarlet(rest)
        # print(f"conjun: arg1:{arg1}, arg2:{arg2}, rest:{rest})")
        return f"&({arg1}, {arg2})", rest
    elif formula.startswith('|'):
        # print("disjunction")
        arg1, rest = prefix_LTL_to_scarlet(formula[2:])
        arg2, rest = prefix_LTL_to_scarlet(rest)
        # print(f"disjun: arg1:{arg1}, arg2:{arg2}, rest:{rest})")
        return f"|({arg1}, {arg2})", rest
    elif formula.startswith('i'):
        ##print("implication")
        arg1, rest = prefix_LTL_to_scarlet(formula[2:])
        arg2, rest = prefix_LTL_to_scarlet(rest)
        # print(f"implic: arg1:{arg1}, arg2:{arg2}, rest:{rest})")
        return f"->({arg1}, {arg2})", rest
    elif formula.startswith('e'):
        # print("equivalence")
        arg1, rest = prefix_LTL_to_scarlet(formula[2:])
        arg2, rest = prefix_LTL_to_scarlet(rest)
        # print(f"equiv: arg1:{arg1}, arg2:{arg2}, rest:{rest})")
        return f"<->({arg1}, {arg2})", rest
    elif formula.startswith('U'):
        # print("until")
        arg1, rest = prefix_LTL_to_scarlet(formula[2:])
        arg2, rest = prefix_LTL_to_scarlet(rest)
        # print(f"until: arg1:{arg1}, arg2:{arg2}, rest:{rest})")
        return f"U({arg1}, {arg2})", rest
    elif formula.startswith('c'):
        # print("symbol")
        if len(formula) > 2:
            # print("symbol rest: ", formula[3:])
            return formula[:2], formula[3:]
        else:
            # print("symbol rest: ")
            return formula[:2], ""
'''
stl_declare = []
from Declare_formulas import formulas, formulas_names
for i in range(len(formulas)):
    fn = formulas_names[i]
    f = formulas[i]
    print("--------------------------{}-------------------------------".format(fn))
    print(i+1)
    print("Original formula: ",f)
    pf = infix_to_prefix(f)
    print("in prefix format: ", pf)
    stl, _ = prefix_LTL_to_STL(pf)
    stl_declare.append(stl)
    print("translated in STL: ", stl)

    print("\n\n")
'''
#formula dataset
#15 and 19 and 1
#existence(c0,2) and chain_succession(c0, c1) and init(c0)