# -*- coding: utf-8 -*-
"""
    Module that tries to optimize valid functions with the polyhedral model using PoCC as an
    optimization tool
"""
from configparser import ConfigParser
from dis import get_instructions
from copy import deepcopy
import os
from subprocess import CalledProcessError
import subprocess, ipdb
import re

from condition import Condition
from loop import Loop
from polynom import Polynom
from scoplib import ScopLib
from statement import Statement
from tree import Tree
from variable import Variable


def __code_from_condition(condition, iterator, is_start):
    """
        Returns the code corresponding to a given condition of a loop statement

    Parameters
    ----------
        condition : the condition_text to convert to code
        iterator : the iterator of the loop this condition_text resides on
        is_start : True if this condition_text is from the start part of the loop and False if it is
            from the end part

    Returns
    -------
        String containing the code of the condition_text
    """
    code = ''
    aux = []
    divisor = 1
    for p in condition.coefficients.keys():
        if p == iterator:
            if (condition.coefficients[p] == 1 and is_start) or (condition.coefficients[p] == -1 and not is_start):
                pass
            else:
                divisor = abs(condition.coefficients[p])
        elif condition.coefficients[p] != 1:
            aux.append(str(p) + ' * ' + str(condition.coefficients[p]))
        else:
            aux.append(str(p))
    code += ' + '.join(aux)
    if condition.term is not None and condition.term != 0:
        if condition.term > 0 and code != '':
            code += ' + ' + str(condition.term)
        else:
            code += str(condition.term)
    if divisor != 1:
        code = '(' + code + ')//' + str(divisor)
    return code


def __get_loop_code(iterator, start_conditions, end_conditions, invert, indentation):
    """
        Returns the code corresponding to a loop

    Parameters
    ----------
    iterator : name of the iterator variables
    start_conditions : list of conditions from the start part of the loop
    end_conditions  : list of conditions from the end part of the loop
    invert : integer that indicate de index of the max or min
    indentation : number of identations this loop has

    Returns
    -------
        A string corresponding to the loop statement described by the inputs

    Raises
    ------
        ValueError: if the iterator is not provided
    """

    code = 'for {} in range ({}{}):\n'
    if iterator is None:
        raise ValueError('Missing iterator')
    if not end_conditions:
        return '\n'

    # Start
    if not start_conditions:
        loop_start = ''
    elif len(start_conditions) == 1:
        loop_start = __code_from_condition(start_conditions[0], iterator, True)
    else:
        aux = []
        for x in start_conditions:
            condition = __code_from_condition(x, iterator, True)
            if condition != '':
                aux.append(condition)
            else:
                aux.append('0')
        if len(aux) > 1:
            arr_in = ['max', 'min']
            loop_start = arr_in[invert[0]] + '({})'.format(' , '.join(aux))
        else:
            loop_start = aux[0]

    # Separator between start and end
    if not (loop_start == '' or loop_start.isspace()):
        loop_start += ' , '

    # End
    if len(end_conditions) == 1:
        condition = __code_from_condition(end_conditions[0], iterator, False)
        if (condition != ''):
            condition = '(' + condition + ')+1'
        loop_end = condition
    else:
        aux = []
        for condition in end_conditions:
            condition = __code_from_condition(condition, iterator, False)
            if condition != '':
                aux.append('(' + condition + ')+1')
            else:
                aux.append('0')
        if len(aux) > 1:
            arr_in = ['max', 'min']
            loop_end = arr_in[invert[0]] + '({})'.format(' , '.join(aux))
        else:
            loop_end = aux[0]

    return indentation * '    ' + code.format(iterator, loop_start, loop_end)


def __get_if_statement_condition(conditions, indentation):
    """
        Returns the code corresponding to a if condition_text with the conditions provided
    Parameters
    ----------
        conditions : list of conditions that make the if statement
        indentation : indentation of the condition_text

    Returns
    -------
        The code corresponding to an if condition_text
    """
    code = indentation * '    ' + 'if({}):\n'
    processed_conditions = []
    for condition in conditions:
        processed_conditions.append('(' + condition.string_form() + ('>= 0)' if condition.greater_than else ' == 0)'))
    result = ' and '.join(processed_conditions)
    return code.format(result)


def __get_statement_code(statement, conditions, indentation):
    """
        Generates the code corresponding to a given statement

    Parameters
    ----------
        statement : statement object to convert to code
        conditions  : list of conditions that affect the statement. If there are any, the statement
            is enclosed inside an if statement.
        indentation : number of indentations of this statement

    Returns
    -------
        A string containing the code corresponding to this statement
    """
    code = ''
    if conditions:
        code += __get_if_statement_condition(conditions, indentation)
        indentation += 1

    code += indentation * '    ' + statement.code[:-1] + '\n'  # -1 to get rid of the ; that PoCC inserts
    return code


def __calculate_all_conditions(tree, possible_coefficients):
    """
        For each node of the tree, it floats all the conditions that are common to ALL the
        children to the parent, removing them from the children
    Parameters
    ----------
        tree : the tree to calculate the conditions
        possible_coefficients : list of all the parameters and the names of all the fathers

    Returns
    -------
        The tree with the conditions recalculated
    """
    if tree.branches:
        for branch in tree.branches:
            __calculate_all_conditions(branch, possible_coefficients + [tree.name])

        conditions = []
        for branch in tree.branches:
            for condition in branch.conditions:
                for branch_2 in tree.branches:
                    if condition not in branch_2.conditions:
                        break
                else:
                    for condition_2 in conditions:
                        if condition == condition_2:
                            break
                    else:
                        for key in condition.coefficients.keys():
                            if key not in possible_coefficients + [tree.name]:
                                break
                        else:
                            conditions.append(condition)
        tree.conditions = conditions
        for branch in tree.branches:
            for condition in tree.conditions:
                branch.conditions.remove(condition)
    else:
        for loop in tree.data.loops:
            tree.conditions += loop.loop_conditions
        for condition in tree.data.if_conditions:
            tree.conditions.append(condition)

    return tree

def __get_statements_pending(tree):
    """
        Searchs all the statements that pends below a tree.
    Parameters
    ----------
        tree : Tree object
    Returns
    -------
        The statements found      
    """
    result = []
    if tree.branches:
        for branch in tree.branches:
            res = __get_statements_pending(branch)
            result += res
        return result
    else:
        return [tree]


def __get_comparation_of_conditions(cond1, cond2):
    """
        Compares two conditions
    Parameters
    ----------
        cond1 : first condition of the loop
        cond2 : second condition of the loop
    Returns
    -------
        If the two conditions are in the form "greater than" returns the index of the greater condition.
        If they are not in this form, returns the index of the least condition.      
    """
    if set(cond1.coefficients) == set(cond2.coefficients):
        res = sum(cond2.coefficients.values()) - sum(cond1.coefficients.values())
        res += cond2.term  - cond1.term 
        if res > 0:
            return 0
        return 1

    else: return -1


def __get_number_of_iterations(cond1, cond2):
    """
        Calculates the number of iterations of a loop with two conditions
    Parameters
    ----------
        cond1 : first condition of the loop
        cond2 : second condition of the loop
    Returns
    -------
        Integer with the value of the iterations    
    """
    if set(cond1.coefficients) == set(cond2.coefficients):
        res = sum(cond2.coefficients.values()) - sum(cond1.coefficients.values())
        if res == 0:
            return abs(cond2.term  - cond1.term)
    return -1


def __get_iteration(cond, iterator):
    """
        Creates a string that represents a loop condition
    Parameters
    ----------
        cond : object Condition
        iterator : name of the iterator of the loop
    Returns
    -------
        String that represents the loop condition
    """
    res = ''
    coef = cond.coefficients
    sign = coef[iterator] *-1
    for k in coef:
        if k!=iterator:
            value = coef[k] * sign
            if (value==1):            
                res+='+' + k
            else:
                res+= '+(' + str(value) + '*' + k +')'
    term = cond.term * sign
    if term>0:
        res+= '+' + str(term)
    elif term<0:
        res+= str(term)
    if res != '':
        res = res[1:] if res[0]=='+' else res
    return res


def __evaluate_set(set, code, iterator):
    """
        Probes if a set is a valid set of conditions to create a loop
    Parameters
    ----------
        set : the two conditions that form a set
        code : the code below the conditions
        iterator : name of the iterator of the loop
    Returns
    -------
        Tuple with the result of the evaluation
    """
    if len(set)==2:
        s1 = set[0]
        s2 = __invert_cond([set[1]], 0)[0]
        comp = __get_comparation_of_conditions(s1, s2)
        if comp!=-1:
            if comp == 1:
                n = __get_number_of_iterations(s1, s2)
                if n==0:
                    it = __get_iteration(s1, iterator)
                    return [code.replace(iterator, it) for _, code in enumerate(code)], True
            else: return None, None
    return code, False


def __modify_code_statements(tree, s1, s2, code):
    """
        Changes the code of two statementa with the indicated string
    Parameters
    ----------
        tree : Tree object where the statements will be found
        s1 : first statement
        s2 : second statement
        code : string with the code to modify
    Returns
    -------
        Tree object modify
    """

    for branch in tree.branches:
        if branch.branches:
            __modify_code_statements(branch, s1, s2, code)
        elif branch.name == s1.name or branch.name == s2.name:
                branch.data = deepcopy(branch.data)
                c = code[0] if branch.name == s1.name else code[1]
                branch.data.code = c

def __invert_cond(list_cond, rest):
    """
        Changes the sign of a list of condition
    Parameters
    ----------
        list_cond : list of conditions
        rest : integer value that gives the option to modify the term condition

    Returns
    -------
        Conditions changed
    """
    
    res = []
    for cond in list_cond:
        c = deepcopy(cond)
        c.coefficients = {k: c.coefficients[k] * -1 for k in c.coefficients}
        c.term += rest
        c.term *= -1
        res.append(c)
    return res



def __mount_pending_branches(sets, s1, s2, s1_n, s2_n, tree, branch):
    """
        Inserts in a branch of a tree the different condition sets
    Parameters
    ----------
        sets : list of conditions set
        s1 : first statement
        s2 : second statement
        s1_n : name representing the first statement in the tree
        s2_n : name representing the second statement in the tree 
        tree : Tree object where are going to insert the conditions
        branch : Tree object child of tree where there are the affected statements 

    Returns
    -------
        Tree object modify
    """
    for s, set in enumerate(sets):
        if s!=2:
            if s%3==0:
                code, one = __evaluate_set(set, [s1.code], branch.name)
            else:
                code, one = __evaluate_set(set, [s2.code], branch.name)
        else: code, one = __evaluate_set(set, [s1.code, s2.code], branch.name)

        if code!=None:
            i = tree.branches.index(branch)
            if not one:
                new = Tree()
                new.name = branch.name
                new.scat = branch.scat + str(i)
                new.conditions = deepcopy(set)
                invert = [1, 1] if s==2 else [0, 0] 
                new.invert = invert
                tree.branches.insert(i, new)
            else:
                new = tree

            if s!=2:
                st = s1 if s%3==0 else s2
                found = False
                first = True
                for _,loop in enumerate(st.loops):
                    found = loop.iterator == branch.name if not found else found
                    if found and loop.iterator != branch.name:
                        newb = Tree()
                        newb.name = loop.iterator
                        newb.conditions = deepcopy(loop.loop_conditions)
                        if first:
                            newb.scat = branch.scat + str(i) if one else '0'
                            first = False
                        new.branches.insert(i, newb)
                        new = newb
                if one: 
                    st = deepcopy(st)
                    st.code = code[0]
                leaf = Tree()
                leaf.name = s1_n if s%3==0 else s2_n
                leaf.data = st
                new.branches.insert(i, leaf)
            else:
                if one:
                    for br in branch.branches:
                        new.insert(i, br)
                        i+=1
                    __modify_code_statements(branch, s1, s2, code)
                else:
                    new.branches = branch.branches
    tree.branches.remove(branch)
            
                
def __calculate_sets_conditions_loops(tree):
    """
        Searchs if exists two statements behind a loop that have different loop 
        conditios. In this cases modify the tree to separete the statements.
    Parameters
    ----------
        tree : Tree object to analyze

    Returns
    -------
        Tree object modify
    
    Raises
    -------
        ValueError in case there are more than two statement that are behind the same loop
        and have different conditions
    """
    for branch in tree.branches:
        if branch.branches:
            if branch.diff:
                statements = []
                statements = __get_statements_pending(branch)
                if len(statements)==2:
                    s1 = statements[0].data
                    s2 = statements[1].data
                    cond_s1 = [loop.loop_conditions for loop in s1.loops if loop.iterator == branch.name][0]
                    cond_s2 = [loop.loop_conditions for loop in s2.loops if loop.iterator == branch.name][0]
                    conds = [cond_s1, cond_s2]
                    ind_st = __get_comparation_of_conditions(cond_s1[0], cond_s2[0])
                    ind_end = __get_comparation_of_conditions(cond_s1[1], cond_s2[1])    

                    if ind_st != -1:
                        start_cond = [conds[ind_st][0]]
                        s1_set1_end = __invert_cond(start_cond, 1)
                        s2_set1_end = s1_set1_end
                    else:
                        start_cond = [conds[0][0], conds[1][0]]
                        s1_set1_end = __invert_cond([cond_s2[0]], 1) + [cond_s1[1]]
                        s2_set1_end = __invert_cond([cond_s1[0]], 1) + [cond_s2[1]]

                    if ind_end != -1:
                        end_cond = [conds[ind_end][1]]
                        s1_set2_st = __invert_cond(end_cond, 1)
                        s2_set2_st = s1_set2_st
                    else:
                        end_cond = [conds[0][1], conds[1][1]]
                        s1_set2_st = [cond_s1[0]] + __invert_cond([cond_s2[1]], 1)
                        s2_set2_st = [cond_s2[0]] + __invert_cond([cond_s1[1]], 1)

                    s1_set1 = [cond_s1[0]] +  s1_set1_end
                    s2_set1 = [cond_s2[0]] +  s2_set1_end
                    intersect_set = start_cond + end_cond
                    s1_set2 = s1_set2_st + [cond_s1[1]]
                    s2_set2 = s2_set2_st + [cond_s2[1]]
                    sets = [s1_set1, s2_set1, intersect_set, s1_set2, s2_set2]
                    __mount_pending_branches(sets, s1, s2, statements[0].name, statements[1].name, tree, branch)
                else:
                    raise ValueError('Not supported')
            __calculate_sets_conditions_loops(branch)


def __is_minor_scat(scat1, scat2):
    """
        Compares two scattering to indicate if scat1 is minor to scat2
    Parameters
    ----------
        scat1 : first scattering list
        scat2 : second scattering list

    Returns
    -------
        True is scat1 is minor scat2
    """
    l_s1 = len(scat1)
    l_s2 = len(scat2)
    if l_s1 != l_s2:
        (minor_len, minor_scat) = (l_s1, scat1) if l_s1 < l_s2 else (l_s2, scat2)
        for i in range(minor_len):
            if scat1[i] != scat2[i]:
                return int(scat1[i]) < int(scat2[i])
        return False
    else: return int(scat1)<int(scat2)



def __tree_from_statements(statements, parameters):
    """
        Creates a tree from the statements, where the leaves are the statements and the nodes are
        the loops, except the top node that is the base level (no scope)
    Parameters
    ----------
        statements : list of statements that will form the leaves
        parameters : list of all the parameters of the scop

    Returns
    -------
        The tree object made from all the statements
    """

    tree = Tree()
    tree.name = 'AuxiliarScope'

    for st, statement in enumerate(statements):
        t = tree
        for sc, scat in enumerate(statement.scattering):     
            same = False
            i=0
            for i in range(len(t.branches)):
                if (scat[1] != t.branches[i].scat):
                    if(__is_minor_scat(scat[1], t.branches[i].scat)): break
                else:
                    n = statement.scattering[sc + 1][0] if sc < len(statement.scattering)-1 else None
                    if n != None:
                        if n == t.branches[i].name:
                            same = True
                            break
                if (i == len(t.branches)-1): i += 1
        
            if scat is statement.scattering[-1]:
                leaf = Tree()
                leaf.data = statement
                leaf.name = 'statement ' + str(st + 1)
                leaf.conditions = statement.if_conditions
                leaf.scat = scat[1]
                t.branches.insert(i, leaf)    
            else:
                if not same:
                    new = Tree()
                    new.name = statement.scattering[sc + 1][0]
                    new.scat = scat[1]
                    t.branches.insert(i, new)
  
                for _, cond in enumerate(statement.loops[sc].loop_conditions):
                    if cond not in t.branches[i].conditions:
                        t.branches[i].conditions.append(cond)
                        if same: t.branches[i].diff = True

            t = t.branches[i]
    __calculate_sets_conditions_loops(tree)
    return tree


def __get_statements_father(tree):
    """
        Searchs recursively all the statements pending below a tree and their fathers in case 
        they are not the frist tree passed on as parameter
    Parameters
    ----------
        tree : tree object 

    Returns
    -------
        The list of statements and their father pending
    """
    res = []
    for branch in tree.branches:
        if branch.branches:
            res_rec = __get_statements_father(branch)
            for r in res_rec:
                r[1] = branch
            res += res_rec
        else:
            res.append([branch, None])
    return res

def __get_statements_to_modify_format(statements, iterator):
    """
        Selects the statements set to modify with the NumPy format related to a determinate loop
    Parameters
    ----------
        statements : list of candidate statements
        iterator : name of the iterator of the loop

    Returns
    -------
        The list of statements set to modify
    """
    res = []

    for st in statements:
        code_s1 = st[0].data.code
        if iterator in code_s1 and 'if' not in code_s1:    #iterator in code
            d = False
            for i in re.finditer(r'(\[[^\[\]]+\])+', code_s1):
                s = i.group()
                if s.count('[') == 1:
                    c = 0
                    for p in s.split(','):
                        if iterator in p:
                            c+=1
                    if c>1 :
                        d = True
                        break
            exp1 = r'(\[[^\[]*' + iterator + r'[^\]]*\]){2}'
            diag = re.findall(exp1, code_s1)
            if not d and not diag:  #not permitted diagonal access
                wp, rp = code_s1.split('=', 1)
                wa = re.search(r'\w+\[', wp)
                wa = wa.group().replace('[', '') if wa else ''
                exp1 = wa + r'(\[[^\[\]]+\])+'
                exp2 = r'\[[^\[]*' + iterator + r'+[^\]]*\]'
                wac = [re.findall(exp2,i.group()) for i in re.finditer(exp1, wp)][0] if wa!='' else None
                rac = [re.findall(exp2,i.group()) for i in re.finditer(exp1, rp)]
                it_dep = False
                if wac:
                    for ac in rac:        #dependence with itself
                        if ac != wac:
                            it_dep = True
                if not it_dep:
                    others_dep = False
                    for st2 in statements:
                        if st2 != st:
                            code_s2 = st2[0].data.code
                            if 'if' not in code_s2:
                                wp_s2, _ = code_s2.split('=', 1)
                                wa_s2 = re.search(r'\w+\[', wp_s2)
                                wa_s2 = wa_s2.group().replace('[', '') if wa_s2 else ''
                                if wa_s2 in rp:  #dependence with others statements
                                    exp1 = wa_s2 + r'(\[[^\[\]]+\])+'
                                    wac_s2 = [re.findall(exp2,i.group()) for i in re.finditer(exp1, wp_s2)][0] if wa_s2!='' else None
                                    rac = [re.findall(exp2,i.group()) for i in re.finditer(exp1, rp)]
                                    if wac_s2:
                                        for ac in rac:
                                            if ac != wac_s2:
                                                others_dep = True
                                    if others_dep:
                                        if st2 in res:
                                            res.remove(st2)
                                        break
                    if not others_dep:
                        res.append(st)
    
    st_to_remove = []
    for st in res:
        for st2 in statements:
            if st!=st2 and st2 not in res:
                code_s1 = st[0].data.code
                code_s2 = st2[0].data.code
                if 'if' not in code_s2:
                    wp_s1, rp_s1 = code_s1.split('=', 1)
                    wp_s2, rp_s2 = code_s2.split('=', 1)
                    if wp_s1 in wp_s2 or wp_s1 in rp_s2 or wp_s2 in rp_s1:
                        st_to_remove.append(st)
                        break
    for st in st_to_remove:
        res.remove(st)
    return res
                        

def __modify_format_statements(statements, conditions, iterator):
    """
        Modify statements with the NumPy format related to a determinate loop
    Parameters
    ----------
        statements : list of statements to modify
        conditions : conditions of the loop 
        iterator : name of the iterator of the loop

    Returns
    -------
        The modify statements
    """
    conditions = deepcopy(conditions)
    min = __get_iteration(conditions[0], iterator)
    conditions[1].term += 1
    max = __get_iteration(conditions[1], iterator)
    min = '0' if min=='' else min
    max = '0' if max=='' else max

    for st in statements:
        code = st[0].data.code
        res = deepcopy(code)
        wp, rp = code.split('=', 1)
        if re.findall(r'\[[^\[\]]+\]', wp) and wp in rp: 
            ind = rp.index(wp)+len(wp)
            if rp[ind]=='+':  #probe if exists +=
                if len(rp) >= ind+5:
                    if rp[ind+1: ind+5] != '(-1*':
                        if iterator not in wp:
                            new_rp = deepcopy(rp)
                            replaces = []
                            for a in re.finditer(r'\w+(\[[^\[\]]+\])+', rp):
                                if iterator in a.group() and a.group() not in replaces:
                                    desc = re.findall(r'\[[^\[\]]+\]', a.group())
                                    ax = 0
                                    if len(desc) < 2:
                                        desc = desc[0].split(',')

                                    for d, dsc in enumerate(desc):
                                        if iterator in dsc:
                                            ax = d
                                            break
                                            
                                    new_rp = new_rp.replace(a.group(), a.group() + '.sum(axis=' + str(ax) + ')')
                                    replaces.append(a.group())
                            res = res.replace(rp, new_rp)

        for a in re.finditer(r'\w+(\[[^\[\]]+\])+', code):
            access = a.group()
            if iterator in access:
                r = '' + access.split('[')[0]
                p = []
                for j in re.finditer(r'\[[^\[\]]+\]', access):
                    ac = j.group().replace('[', '').replace(']', '')
                    if iterator in ac:
                        for part in ac.split(','):
                            if iterator in part:
                                mi = part.replace(iterator, min)
                                ma = part.replace(iterator, max)
                                p.append(mi + ':' + ma)
                            else:
                                p.append(part)
                    else:
                        p.append(ac)
                r+= '[' + ','.join(p) + ']'
                res = res.replace(access, r)
        
        wp, rp = res.split('=', 1)
        if wp.count(':') >1 or (wp.count(':') == 1 and ',' not in wp and '*' in rp and not 'stddev' in wp) and'.sum' in rp:
            for i in range(4):
                res = res.replace('.sum(axis=' + str(i) + ')', '')
        
        sq = re.findall(r'sqrt\([^\)]+\)', rp)
        for s in sq:
            if ':' in s:
                res = res.replace(s, 'np.'+ s)

        st[0].data.code = res

def __insert_statement_tree(childs, statement):
    """
        Inserts a statement in a tree
    Parameters
    ----------
        childs : list of trees that are childs of the tree where is going to insert the statement 
        statement : statement set to insert

    Returns
    -------
        The new list of childs of the tree where the statement was insert  
    """
    for c, child in enumerate(childs):
        if __is_minor_scat(statement.scat, child.scat):
            break
        if statement.scat == child.scat:
            break
        if c==len(childs)-1: c+=1
    childs.insert(c, statement)

def __remove_statement_tree(tree, statement):
    """
        Removes a statement from a tree
    Parameters
    ----------
        tree : Tree object 
        statement : statement set to remove

    Returns
    -------
        The tree object modify  
    """
    for branch in tree.branches:
        if not branch.branches:
            if branch.data == statement:
                tree.branches.remove(branch)
                break
        else:
            __remove_statement_tree(branch, statement)
    


def __modify_tree_with_format_statements(stat_tot, stat_mod, tree, branch):
    """
        Modify a tree part with the modify statements
    Parameters
    ----------
        stat_tot : list of the total statements below the branch
        stat_mod : list part of the total statements with the modify statements
        tree : Tree object father of the branch where the statements are found
        branch : Tree object chid of tree where the statements are found

    Returns
    -------
        The tree object modify 
    """
    d_trees = {}
    tree_created = []
    for st in stat_tot:
        if st[1] is not None:
            if st[1] in d_trees:
                d_trees[st[1]].append(st[0])
            else:
                d_trees[st[1]] = [st[0]]
    for st in stat_mod:
        if st[1] is not None:
            d_trees[st[1]].remove(st[0])
    for st in stat_mod:
        if st[1] == None:
            if not branch.branches.index(st[0]):
                st[0].scat = branch.scat + st[0].scat if branch.scat != '0' else st[0].scat
            tree.branches.insert(tree.branches.index(branch),st[0])
            #__insert_statement_tree(tree.branches, st[0])
            branch.branches.remove(st[0])
        elif not d_trees[st[1]]:
            if st[1] not in tree.branches:
                if not branch.branches.index(st[1]):
                    st[1].scat = branch.scat + st[1].scat if branch.scat != '0' else st[1].scat
                tree.branches.insert(tree.branches.index(branch),st[1])
                #__insert_statement_tree(tree.branches, st[1])
                branch.branches.remove(st[1])
        else:
            if st[1] not in tree_created:
                new = deepcopy(st[1])
                if not branch.branches.index(st[1]):
                    new.scat = branch.scat + new.scat  if branch.scat != '0' else new.scat
                __insert_statement_tree(tree.branches, new)
                tree_created.append(st[1])
                for stat in d_trees[st[1]]:
                    __remove_statement_tree(new, stat.data)
            __remove_statement_tree(st[1], st[0].data)
    if not branch.branches:
        tree.branches.remove(branch)



def __format_tree_numpy(tree):
    """
        Tries recursively to apply the NumPy format to a tree representing a code 
    Parameters
    ----------
        tree : Tree object representing the tree to process

    Returns
    -------
        The tree object modify 
    """
    
    for branch in tree.branches:
        if branch.branches:
            for sub_branch in branch.branches:
                if sub_branch.branches:
                    __format_tree_numpy(branch)
            
            stat_tot = []
            stat_mod = []
            stat_tot = __get_statements_father(branch)
        
            stat_mod = __get_statements_to_modify_format(stat_tot, branch.name)

            if len(branch.conditions) == 2:
                __modify_format_statements(stat_mod, branch.conditions, branch.name)
                __modify_tree_with_format_statements(stat_tot, stat_mod, tree, branch)

def __analize_op_dot(op1, op2):
    """
        Analizes the corresponding np.dot multiply transformation to do and modifies the operands to adapt it
    Parameters
    ----------
        op1 : string representing the first multiplication operand
        op2 : string representing the second multiplication operand

    Returns
    -------
        The operands modifies with the needed transformation 
    """
    sp1 = re.search(r'\[[^\[]*[^\]]*\]', op1).group().replace('[','').replace(']', '').split(',')
    sp2 = re.search(r'\[[^\[]*[^\]]*\]', op2).group().replace('[','').replace(']', '').split(',')

    if len(sp1)==2 and len(sp2)==2:
        if len(sp1[0])==1 or len(sp1[1])==1 and len(sp2[0])==1 or len(sp2[1])==1:
            rang1 = sp1[1] if len(sp1[0])==1 else sp1[0]
            rang2 = sp2[1] if len(sp2[0])==1 else sp2[0]
            if rang1 != rang2:
                op1 = op1.replace(']', ',np.newaxis]')
                op2 = op2.replace('[', '[np.newaxis,')
                return op1, op2
    if len(sp1)==2 and len(sp2)==1:
        if sp1[1] != sp2[0]:
            op1 = op1.replace(',' + sp1[1] + ']', '].T')

    return op1, op2


def __format_multiplications_with_numpy_dot(tree):
    """
        Tries to modify simple multiplications with the NumPy function np.dot
    Parameters
    ----------
        tree : Tree object representing the tree to process

    Returns
    -------
        The tree object modify 
    """
    statements = __get_statements_pending(tree)
    for st in statements:
        code = st.data.code
        if not '.sum' in code:
            for m in re.finditer(r'\w+(\[[^\[\]]+\])+[\)]?\*\w+(\[[^\[\]]+\])+', code):
                s = m.group()
                if ')' in s:
                    l = list(code)
                    for i in range(code.index(s), -1, -1):
                        if l[i] == '(':
                            l[i] = ''
                            break
                    code = ''.join(l)
                    s = s.replace(')', '')
                op1, op2 = s.split('*')
                if len(re.findall(r'\[[^\[\]]+\]', op1))== 1 and len(re.findall(r'\[[^\[\]]+\]', op2))==1:
                    op1, op2 = __analize_op_dot(op1, op2)
                    r = 'np.dot(' + op1 + ',' + op2 + ')'
                    st.data.code = code.replace(m.group(), r)
        else:
            
            _, rp = code.split('=', 1)
            for cor in re.findall(r'\[[^\[\]]+\]', rp):
                rp = rp.replace(cor, cor.replace('+', '$'))
            for mul in re.finditer(r'[^\+]+\*[^\*\+]+',rp):
                or_mul = mul.group().replace('$', '+')
                ch_mul = deepcopy(or_mul)
                axs = re.findall(r'axis=[0-9]', or_mul)
                if len(axs) > 1:
                    same = True
                    for i in range(len(axs)):
                        if axs[0] != axs[i]:
                            same = False
                            break
                    if same:
                        ax_ch = deepcopy(axs[0])
                        if ch_mul.count('*')==1:
                            l, r = ch_mul.split('*') 
                            if '[' in l and '[' in r:
                                sp1 = re.search(r'\[[^\[]*[^\]]*\]', l).group().replace('[','').replace(']', '').split(',')
                                sp2 = re.search(r'\[[^\[]*[^\]]*\]', r).group().replace('[','').replace(']', '').split(',')
                                if len(sp1)==2 and len(sp2)==2:
                                    if len(sp1[0])!=1 and len(sp1[1])==1 and len(sp2[0])!=1 and len(sp2[1])!=1:
                                        if sp1[0] == sp2[0] and sp1[1] in sp2[1]:
                                            ch_mul = ch_mul.replace(r, r.replace(']', '].T'))
                                            axs[0] = 'axis=1'
                        ch_mul = ch_mul.replace('.sum(' + ax_ch + ')', '')
                        l = list(ch_mul)
                        ref = ch_mul.count('(')
                        cont = 0
                        for c, car in enumerate(l):
                            if car == ')':
                                cont+=1
                            if cont == ref:
                                break
                        l.insert(c+1, '.sum(' + axs[0] + ')')
                        ch_mul = ''.join(l)
                        st.data.code = st.data.code.replace(or_mul, ch_mul)


def __get_numpy_format(tree):
    __format_tree_numpy(tree)
    __format_multiplications_with_numpy_dot(tree)


def __get_code(tree, code='', indentation=0):
    """
        Returns a string that contains the code described by a given tree.
    Parameters
    ----------
        tree : a tree object that describes a code
        code : used in recursion (default = '')
        indentation : indentation of the code (default = '')

    Returns
    -------
        A string of containing the code described by the tree
    """
    if tree.branches:  # not a statement
        if tree.conditions:
            start_conditions = []
            end_conditions = []
            if_conditions = []
            iterator = tree.name
            invert = tree.invert
            if tree.name == 'AuxiliarScope':
                code += __get_if_statement_condition(tree.conditions, indentation + 1)
                indentation += 1
                for branch in tree.branches:
                    code = __get_code(branch, code, indentation + 1)
            else:
                for condition in tree.conditions:
                    if tree.name in condition.coefficients.keys():  # The condition_text is from the loop, not an if
                        if condition.coefficients[tree.name] >= 0:
                            for key in condition.coefficients.keys():  # Start conditions's coefficients are inverted
                                if key != iterator:
                                    condition.coefficients[key] *= -1
                            condition.term *= -1
                            start_conditions.append(condition)
                        else:
                            #condition.term += 1  # Range is <, while the condition_text is <=
                            end_conditions.append(condition)
                    else:
                        if_conditions.append(condition)
                if if_conditions:
                    code += __get_if_statement_condition(if_conditions, indentation)
                    indentation += 1
                code += __get_loop_code(iterator, start_conditions, end_conditions, invert, indentation)
                for branch in tree.branches:
                    code = __get_code(branch, code, indentation + 1)
        else:
            for branch in tree.branches:
                code = __get_code(branch, code, indentation + 1)
    else:
        code += __get_statement_code(tree.data, tree.conditions, indentation)

    return code


def __calculate_range_of_iterations(code_block, parameters, var_id_by_var_name, number, iterator):
    """
        Creates a loop object based on the block of bytecode instructions given. Any new parameter
        will be added to the parameter list. If any new variables appear, the number will be
        incremented and added to the dictionary of variables
    Parameters
    ----------
        code_block : list of bytecode instructions corresponding to a for instruction. From the
            SETUP_LOOP to the line before FOR_LOOP
        parameters : the list of parameters of the scop
        var_id_by_var_name : dictionary associating variable_string names with their corresponding
            number
        number : the max number already associated to a variable_string inside a list so it
            becomes mutable
        iterator : name of the iterator

    Returns
    -------
        A new instance of Loop with the information extracted from the block

    Raises
    ------
        ValueError if the step is different to 1, because SCoPLib doesn't accept different
            values to 1
    """
    start_conditions = []
    end_conditions = []
    step = 1

    stack = []
    self = 0
    for i,line in enumerate(code_block):
        if line.opname == 'FOR_ITER':
            pass
        elif line.opname == "JUMP_ABSOLUTE":
            pass
        elif line.opname == 'GET_ITER':
            pass
        elif line.opname == 'LOAD_GLOBAL':
            stack.append(line.argval)
        elif (line.opname == 'LOAD_CONST') or (line.opname == "LOAD_ATTR"):
            if self:
                s = 'self.' + str(line.argval)
                self=0
            else: s = str(line.argval)
            stack.append(s)
        elif line.opname == 'LOAD_FAST':
            if line.argval == "self": 
                self = 1
                continue
            stack.append(line.argval)
            if '|' + str(line.argval) + '|' not in var_id_by_var_name:
                parameters.append(str(line.argval))
                number[0] += 1
                var_id_by_var_name['|' + str(line.argval) + '|'] = number[0]
        elif line.opname == 'BINARY_SUBSCR':
            tos = stack.pop()
            tos1 = stack.pop()
            stack.append(tos1[tos])
        elif line.opname == 'BINARY_ADD':
            tos = stack.pop()
            tos1 = stack.pop()
            stack.append(tos + '|+|' + tos1)
        elif line.opname == 'BINARY_SUBTRACT':
            tos = stack.pop()
            tos1 = stack.pop()
            stack.append(tos1 + '|-|' + tos)
        elif line.opname == 'BINARY_MULTIPLY':
            tos = stack.pop()
            tos1 = stack.pop()
            stack.append(tos + '|*|' + tos1)
        elif line.opname == 'CALL_FUNCTION':
            args = []
            for _ in range(int(line.argval)):
                args.append(stack.pop())
            tos = stack.pop()
            stack.append(str(tos) + '(' + str(args) + ')')
            if str(tos) == 'range':
                if len(args) >= 1:
                    if 'min' not in args[0] and 'max' not in args[0]:
                        polynom = __analyze_polynom(args[0], [], parameters, [-1], var_id_by_var_name)
                        condition = Condition()
                        condition.term = polynom.term
                        condition.coefficients = polynom.coefficients
                        condition.coefficients['|' + iterator + '|'] = -1
                        end_conditions.append(condition)
                if len(args) >= 2:
                    if 'max' not in args[1] and 'min' not in args[1]:
                        polynom = __analyze_polynom(args[1], [], parameters, [-1], var_id_by_var_name)
                        condition = Condition()
                        condition.term = polynom.term
                        condition.coefficients = polynom.coefficients
                        condition.coefficients['|' + iterator + '|'] = 1
                        start_conditions.append(condition)
                else:
                    condition = Condition()
                    condition.coefficients['|' + iterator + '|'] = 1
                    start_conditions.append(condition)  # If not conditions on the start, a 0 will be added by default
                if len(args) >= 3:
                    step = args[2]
                else:
                    step = 1

            elif str(tos) == 'len':
                raise ValueError('len operation inside range arguments')
            elif str(tos) == 'max':
                for arg in args:
                    polynom = __analyze_polynom(arg, [], parameters, [-1], var_id_by_var_name)
                    condition = Condition()
                    condition.term = polynom.term
                    condition.coefficients = polynom.coefficients
                    start_conditions.append(condition)
            elif str(tos) == 'min':
                for arg in args:
                    polynom = __analyze_polynom(arg, [], parameters, [-1], var_id_by_var_name)
                    condition = Condition()
                    condition.term = polynom.term
                    condition.coefficients = polynom.coefficients
                    end_conditions.append(condition)
        else:
            print("DEBUG: argument not considered", line.opname)
            for x in code_block:
                print("DEBUG: code block: ", x)
            raise ValueError('DEBUG: value not considered')

    # All coefficients from the start will be inverted except for the corresponding to the iterator.
    for condition in start_conditions:
        for coefficient in condition.coefficients.keys():
            if coefficient != '|' + iterator + '|':
                condition.coefficients[coefficient] *= -1
        condition.term *= -1

    # The terms of the end condition_text will be subtracted 1 because range uses < while SCoPLib uses <=
    for condition in end_conditions:
        condition.coefficients['|' + iterator + '|'] = -1
        condition.term -= 1

    loop = Loop()
    loop.loop_conditions += start_conditions
    loop.loop_conditions += end_conditions
    if step != 1:
        raise ValueError('Step must be 1')
    return loop


def __analyze_polynom(polynom_string, loops, parameters, number, var_id_by_var_name):
    """
        Given a string of a polynom, it returns the polynom object that represents the same string.
            If a new parameter is found in the polynom_string it will be added to the parameter
            list. Any new variables found will be added to the dictionary and number will be
            incremented accodingly

    Parameters
    ----------
        polynom_string : string of a polynom where all the values are tokenized. i.e. inside | |
        loops : list of all the loops under which this polynom resides.
        parameters : list of parameters
        number : the max number already associated to a variable_string inside a list so it
            becomes mutable
        var_id_by_var_name : dictionary associating variable_string names with their corresponding
            number

    Returns
    -------
        A polynom object that represents the input polynom
    """
    pol = Polynom()
    sums = polynom_string.split('|+|')
    coefficients = {}
    for product in sums:
        product = product.replace('(', '').replace(')', '')
        try:  # term case
            term = eval(product.replace('|', ''))
            pol.term += term
            continue
        except (NameError, SyntaxError):
            # An error indicates that the product is different of a single number, so we know its a product
            pass
        elements = product.split('|*|')
        name = None
        # On a list for the case of the value appearing before the variable_string, the value will be maintained
        coefficient = [1]
        for element in elements:
            element = element.replace('|', '')
            try:
                c = int(element)
                coefficient[0] *= c
                continue
            except ValueError:
                for loop in loops:
                    if loop.iterator.replace('|', '') == element:
                        name = loop.iterator
                        break
                if name is not None:
                    coefficients['|' + name + '|'] = coefficient
                else:
                    if element not in parameters:
                        if '|' + str(element) + '|' not in var_id_by_var_name.keys():
                            # Iterator variables are assigned a number and added to the dictionary
                            number[0] += 1
                            var_id_by_var_name['|' + str(element) + '|'] = number[0]
                        parameters.append(element)
                    coefficients['|' + str(element) + '|'] = coefficient

                continue

    for x in coefficients.keys():  # Coefficients are taken out of the list since now its not necessary
        pol.coefficients[x] = coefficients[x][0]

    return pol


def __analyze_var(variable_string, var_id_by_var_name, loops, number, parameters):
    """
        Given a string of a variable, it returns the variable object that represents the same
            string. If a new parameter is found in the variable_string it will be added to the
            parameter list. Any new variables found will be added to the dictionary and number
            will be incremented accordingly

    Parameters
    ----------
        variable_string : string of a variable where all the values are tokenized. i.e. inside | |
        var_id_by_var_name : dictionary associating variable_string names with their corresponding
            number
        loops : list of all the loops under which this polynom resides
        number : the max number already associated to a variable_string inside a list so it becomes
            mutable
        parameters : list of parameters
    Returns
    -------
        A variable object that represents the input polynom
    """
    result_variables = []

    if '|,|' in variable_string:  # When the variable is an argument of a function, it must be split
        variables = variable_string.split('|,|')
    else:
        variables = [variable_string]

    for variable_string in variables:
        try:  # If the variable is a number
            float(variable_string.replace('|', '').replace(')', '').replace('(', ''))
            return []
        except ValueError:
            pass

        variable_string = variable_string.replace(']', '').replace('(', '').replace(')', '')
        ls = variable_string.split('[')
        var_name = ls[0]
        indexes = []
        if len(ls) > 1:
            for e in ls[1:]:
                indexes.append(__analyze_polynom(e, loops, parameters, number, var_id_by_var_name))

        is_var_id_known = next((x for x in var_id_by_var_name.keys() if x == var_name), False)
        if not is_var_id_known:
            number[0] += 1
            result_variable = Variable(var_name, number[0])
            var_id_by_var_name[var_name] = number[0]
        else:
            result_variable = Variable(var_name, var_id_by_var_name[var_name])

        result_variable.index = indexes

        if indexes:  # Case when the variable is a list of a matrix
            result_variable.index_text = '[' + ']['.join(ls[1:]) + ']'
        else:  # Case that the variable doesn't have an index
            result_variable.index_text = '[0]'
        result_variables.append(result_variable)

    return result_variables


def __separate_vars(var_string):
    """
        Separates a variable with the form a[x]+b[y] in [a[x], b[y]], returning them as a list

    Parameters
    ----------
     var_string : string of a sum of variables

    Returns
    -------
        A list of the variables separated
    """
    analyzed_string = ''
    in_variable = False
    for i, char in enumerate(var_string):
        if char == '[':
            in_variable = True
        elif char == ']':
            in_variable = False
        elif char in ('+', '/', '*', '-', '>', '<', '>=', '<=') and not in_variable and var_string[i + 1] == '|':
            # Followed by a '|' to avoid the - in negative number
            char = '#'
        analyzed_string += char
    # This makes the ',' added in CALL_FUNCTIONS treated. The ',' is necessary to build the body
    analyzed_string.replace(',', '#')

    return analyzed_string.split('|#|')


def __analyze_condition(condition_text, loops, parameters, number, var_Id_by_varName):
    """
        Generates a condition object based on a tokenized string representing a condition
            Parameters
    ----------
        condition_text : string of a condition where all the values are tokenized. i.e. inside | |
        loops : list of all the loops under which this polynom resides
        parameters : list of parameters
        number : the max number already associated to a variable_string inside a list so it becomes
                mutable
        var_Id_by_varName : dictionary associating variable_string names with their corresponding
                number

    Returns
    -------
        A condition object that represents the input condition

    """
    st = ''
    condition_text = condition_text[1:-1]  # Parenthesis are deleted
    if '<' in condition_text:
        if '=' in condition_text:
            left, right = condition_text.split('|<=|')
            lefts = str(left).split('|+|')
            for l in lefts:
                st += '|+||-1||*|' + l
            processed = str(right) + str(st) 
        else:
            left, right = condition_text.split('|<|')
            lefts = str(left).split('|+|')
            for l in lefts:
                st += '|+||-1||*|' + l
            processed = str(right) + str(st) + '|+||-1|'
        is_greater_than = True
        
    elif '>' in condition_text:
        if '=' in condition_text:
            left, right = condition_text.split('|>=|')
            rights = str(right).split('|+|')
            for r in rights:
                st += '|+||-1||*|' + r
            processed = str(left) + str(r)
        else:
            left, right = condition_text.split('|>|')
            rights = str(right).split('|+|')
            for r in rights:
                st += '|+||-1||*|' + r
            processed = str(left) + str(r) + '|+||-1|'
        is_greater_than = True

    else:
        left, right = condition_text.split('|==|')
        is_greater_than = False
        processed = str(left) + '|+||-1||*|' + str(right)

    analyzed = __analyze_polynom(processed, loops, parameters, number, var_Id_by_varName)
    condition = Condition()
    condition.coefficients = analyzed.coefficients
    condition.term = analyzed.term
    condition.greater_than = is_greater_than
    return condition


def __obtain_statements_info(code_block, loops, if_conditions, var_id_by_var_name, number, parameters, scattering):
    """
        Obtains the list of statement objects that appear in the code_block
    Parameters
    ----------
        code_block: list of bytecode instructions of the block of code to analyze
        loops: list of loops under which this scope resides
        if_conditions: list of conditions under which this scop resides
        var_id_by_var_name: dictionary that maps a variable name to its corresponding number
        number: last number asigned to a variable inside a list. It needs to be inside a list so its passed by value
        parameters: list of the names of the parameters for this scope
        scattering: list of tuples of the scattering of the previous statements

    Returns
    -------
        A list of all the statements of the code_block as Statement objects
    """
    loop_start = 0
    # First loop start (there's always at least one)
    for i, line in enumerate(code_block):
        if line.opname == 'FOR_ITER':
            iterator = code_block[i + 1].argval
            loop_info = __calculate_range_of_iterations(code_block[:i - 1], parameters,
                                                        var_id_by_var_name, number, iterator)
            loop_info.iterator = iterator
            loop_info.if_conditions = if_conditions
            if_conditions = []
            if '|' + str(loop_info.iterator) + '|' not in var_id_by_var_name.keys():
                number[0] += 1
                var_id_by_var_name['|' + str(loop_info.iterator) + '|'] = number[0]
            loops = loops + [loop_info]
            if scattering:  # Before adding a new scattering the previous is incremented
                scattering[-1][1] += 1
            scattering.append([loop_info.iterator, -1])
            loop_start = i + 2
            break

    # Se itera todos los statement hasta acabar
    in_loop = False
    loop_stack = 0
    start = 0
    statements = []
    stack = []
    body = []
    self = 0
    exist_else = False
    current_statement = Statement()
    for i, line in enumerate(code_block[loop_start:]):
        # caso de bucles anidados
        if line.opname == 'FOR_ITER':
            if loop_stack == 0:
                start = __find_loop_header_start( code_block[loop_start:], i )
            loop_stack += 1
            in_loop = True
        elif line.opname == 'JUMP_ABSOLUTE':
            loop_stack -= 1
            if loop_stack == 0:
                statements.extend(
                    __obtain_statements_info(code_block[loop_start + start:i + loop_start], loops, if_conditions,
                                             var_id_by_var_name, number, parameters, scattering))
                in_loop = False
        elif line.opname == "GET_ITER":
            continue
        # Caso no es un bucle, por lo que busca statements
        else:
            if not in_loop:
                if line.opname == 'STORE_FAST':
                    tos = str(stack.pop())
                    b = str(body.pop())
                    wrote_var = '|' + str(line.argval) + '|'
                    current_statement.code = str(line.argval) + '=' + b
                    current_statement.wrote_vars += __analyze_var(wrote_var, var_id_by_var_name, loops, number,
                                                                  parameters)
                    for var in __separate_vars(tos):
                        current_statement.read_vars += __analyze_var(var, var_id_by_var_name, loops, number, parameters)
                    if exist_else:
                        current_statement.read_vars += __analyze_var(stack.pop(), var_id_by_var_name, loops, number, parameters)
                        for var in __separate_vars(stack.pop()):
                            current_statement.read_vars += __analyze_var(var, var_id_by_var_name, loops, number, parameters)
                      
                        exist_else = False

                    current_statement.loops = loops.copy()
                    scattering[-1][1] += 1  # Se aumenta en una posicion el scattering actual
                    current_statement.scattering = deepcopy(scattering)
                    current_statement.if_conditions = if_conditions
                    if_conditions = []
                    for loop in current_statement.loops:
                        current_statement.original_iterator_names.append(loop.iterator)

                    statements.append(current_statement)
                    current_statement = Statement()
                elif line.opname == 'STORE_SUBSCR':
                    tos = str(stack.pop())  # Index
                    tos1 = str(stack.pop())  # Varname
                    tos2 = str(stack.pop())
                    stack.append(tos1 + '[' + tos + ']|=|' + tos2)
                    b = str(body.pop())  # Index
                    b1 = str(body.pop())  # Varname
                    b2 = str(body.pop())
                    body.append(b1 + '[' + b + ']=' + b2)
                    if exist_else:
                        st = body.pop()
                        el = body .pop()
                        ifp = body.pop()
                        current_statement.code = st + ifp + el
                        current_statement.wrote_vars += __analyze_var(stack[-1].split('|=|')[0], var_id_by_var_name, loops,
                                                                  number, parameters)
                        for var in __separate_vars(stack[-1].split('|=|')[1]):
                            current_statement.read_vars += __analyze_var(var, var_id_by_var_name, loops, number, parameters)
                        current_statement.read_vars += __analyze_var(stack[-2], var_id_by_var_name, loops, number, parameters)
                        for var in __separate_vars(stack[-3]):
                            current_statement.read_vars += __analyze_var(var, var_id_by_var_name, loops, number, parameters)
                    
                        exist_else = False         
                    else: 
                        current_statement.code = body.pop()
                        current_statement.wrote_vars += __analyze_var(stack[-1].split('|=|')[0], var_id_by_var_name, loops,
                                                                  number, parameters)
                        for var in __separate_vars(stack[-1].split('|=|')[1]):
                            current_statement.read_vars += __analyze_var(var, var_id_by_var_name, loops, number, parameters)
                    current_statement.loops = loops.copy()
                    scattering[-1][1] += 1  # Se aumenta en una posicion el scattering actual
                    current_statement.scattering = deepcopy(scattering)
                    current_statement.if_conditions = if_conditions
                    if_conditions = []
                    for loop in current_statement.loops:
                        current_statement.original_iterator_names.append(loop.iterator)
                    statements.append(current_statement)
                    current_statement = Statement()
                elif line.opname == 'LOAD_FAST':
                    if line.argval != "self":
                        stack.append('|' + format(line.argval) + '|')
                        body.append(format(line.argval))
                    else:
                        self = 1
                elif line.opname == 'LOAD_CONST' or line.opname == "LOAD_ATTR":
                    if self:
                        s = 'self.' + format(line.argval)
                        self = 0
                    else:
                        s = format(line.argval)
                    stack.append('|' + s + '|')
                    body.append(s)
                elif line.opname == 'LOAD_GLOBAL':
                    stack.append(line.argval)
                    body.append(line.argval)
                elif line.opname == 'DUP_TOP':
                    tos = stack.pop()
                    stack.append(tos)
                    stack.append(tos)
                    b = body.pop()
                    body.append(b)
                    body.append(b)
                elif line.opname == 'DUP_TOP_TWO':
                    tos = stack.pop()
                    tos1 = stack.pop()
                    stack.append(tos1)
                    stack.append(tos)
                    stack.append(tos1)
                    stack.append(tos)
                    b = body.pop()
                    b1 = body.pop()
                    body.append(b1)
                    body.append(b)
                    body.append(b1)
                    body.append(b)
                elif line.opname == 'ROT_TWO':
                    tos = stack.pop()
                    tos1 = stack.pop()
                    stack.append(tos)
                    stack.append(tos1)
                    b = body.pop()
                    b1 = body.pop()
                    body.append(b)
                    body.append(b1)
                elif line.opname == 'ROT_THREE':
                    tos = stack.pop()
                    tos1 = stack.pop()
                    tos2 = stack.pop()
                    stack.append(tos)
                    stack.append(tos2)
                    stack.append(tos1)
                    b = body.pop()
                    b1 = body.pop()
                    b2 = body.pop()
                    body.append(b)
                    body.append(b2)
                    body.append(b1)
                elif line.opname == 'BINARY_ADD' or line.opname == 'INPLACE_ADD':
                    tos = str(stack.pop())
                    tos1 = str(stack.pop())
                    stack.append('(' + tos1 + '|+|' + tos + ')')
                    b = body.pop()
                    b1 = body.pop()
                    body.append('(' + b1 + '+' + b + ')')
                elif line.opname == 'BINARY_SUBTRACT' or line.opname == 'INPLACE_SUBTRACT':
                    tos = str(stack.pop())
                    tos1 = str(stack.pop())
                    stack.append('(' + tos1 + '|+|(|-1||*|' + tos + '))')
                    b = str(body.pop())
                    b1 = str(body.pop())
                    body.append('(' + b1 + '+(-1*' + b + '))')
                elif line.opname == 'BINARY_MULTIPLY' or line.opname == 'INPLACE_MULTIPLY':
                    tos = str(stack.pop())
                    tos1 = str(stack.pop())
                    stack.append('(' + tos1 + '|*|' + tos + ')')
                    b = str(body.pop())
                    b1 = str(body.pop())
                    body.append('(' + b1 + '*' + b + ')')
                elif line.opname == 'BINARY_TRUE_DIVIDE' or line.opname == 'INPLACE_TRUE_DIVIDE':
                    tos = str(stack.pop())
                    tos1 = str(stack.pop())
                    stack.append('(' + tos1 + '|/|' + tos + ')')
                    b = str(body.pop())
                    b1 = str(body.pop())
                    body.append('(' + b1 + '/' + b + ')')
                elif line.opname == 'BINARY_SUBSCR':
                    tos = str(stack.pop())  # Index
                    tos1 = str(stack.pop())  # varName
                    stack.append('(' + tos1 + '[' + tos + '])')
                    b = body.pop()
                    b1 = body.pop()
                    body.append( b1 + '[' + b + ']')
                elif line.opname == 'COMPARE_OP':
                    tos = stack.pop()
                    tos1 = stack.pop()
                    stack.append('(' + tos + '|' + str(line.argval) + '|' + tos1 + ')')
                    b = str(body.pop())
                    b1 = str(body.pop())
                    body.append('(' + b + str(line.argval) + b1 + ')')
                elif line.opname == 'UNARY_NEGATIVE':
                    tos = stack.pop()
                    stack.append('(-1*' + tos + ')')
                    b = str(body.pop())
                    body.append('(-1*' + b + ')')
                elif line.opname == 'JUMP_ABSOLUTE' or line.opname == 'JUMP_FORWARD':
                    pass
                elif line.opname == 'POP_BLOCK':
                    pass
                elif line.opname == 'POP_JUMP_IF_TRUE':
                    tos = stack.pop()
                    body.pop()
                    condition = __analyze_condition(tos, loops, parameters, number, var_id_by_var_name)
                    if_conditions.append(condition)
                elif line.opname == 'POP_JUMP_IF_FALSE':
                    exist_else = False
                    for j, l in enumerate(code_block[loop_start + i:]):
                        if l.opname == 'JUMP_FORWARD':
                            exist_else = True
                        elif l.opname == 'JUMP_ABSOLUTE':
                            break
                    tos = stack.pop()
                    b = body.pop()
                    if not exist_else:
                        condition = __analyze_condition(tos, loops, parameters, number, var_id_by_var_name)
                        if_conditions.append(condition)
                    else:
                        stack.append(tos)
                        body.append(' if ' + b + ' else ' )

                elif line.opname == 'CALL_FUNCTION' or line.opname == 'CALL_METHOD':
                    args = []
                    b = []
                    for _ in range(line.argval):
                        args.append(stack.pop())
                        b.append(body.pop())
                    stack.pop()  # funcion a ejecutar
                    args.reverse()
                    b.reverse()
                    stack.append('|,|'.join(args))
                    body.append(str(body.pop()) + '(' + ', '.join(b) + ')')
                elif line.opname == 'LOAD_METHOD':
                    if self:
                        tos = 'self'
                        b = 'self'
                        self = 0
                    else:
                        tos = stack.pop()
                        b = body.pop()
                    stack.append(tos + '.' + line.argval)
                    body.append(b + '.' + line.argval)
                elif line.opname == 'EXTENDED_ARG':
                    pass
                else:
                    print("DEBUG: argument not considered", line.opname)
                    for x in code_block:
                        print("DEBUG: code block: ", x)
                    raise ValueError('DEBUG: value not considered')

    scattering.pop()
    for loop in loops:  # Iterator variables are deleted from the parameters
        if loop.iterator in parameters:
            parameters.remove(loop.iterator)
    return statements


def __optimize_loops(function, numpy_format, debug_mode):
    """
        Tries to optimize a function.
    Parameters
    ----------
        function : the function whose loops have to be optimized
        numpy_format : if True generates the code in NumPy format
        debug_mode : if True the files used in the optimization are not deleted

    Returns
    -------
        An optimized version of the function or the original code if the
        optimization failed

    Raises
    ------
        CalledProcessError : if the optimization fails
    """
    config = ConfigParser()
    config.read('polypy.conf')
    pocc_path = config['DEFAULT']['pocc_path']

    generate_scoplib_file(function, 'polypy')

    # Optimization of SCoPLib

    optimization_process = subprocess.Popen([pocc_path + ' --read-scop --output-scop polypy.scop --pluto'],
                                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    optimization_process.wait()
    optimization_process_output, optimization_process_output_error = optimization_process.communicate()
    if str(optimization_process_output) == 'b\'\'':
        if debug_mode:
            for line in optimization_process_output_error.decode('utf-8').split('\n'):
                print('DEBUG:', line, '\n')
        raise CalledProcessError(1, 'pocc --read-scop --output-scop polypy.scop --pluto from')
    else:
        if debug_mode:
            print('DEBUG: output_scop_output')
            for line in optimization_process_output.decode('utf-8').split('\n'):
                print('DEBUG:', line)
            print('\n\n\n')
    '''
    output_scop_process = subprocess.Popen([pocc_path + ' --output-scop polypy.pocc.c '],
                                           shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output_scop_process.wait()
    output_scop_output, output_scop_error = output_scop_process.communicate()
    if str(output_scop_output) == 'b\'\'':
        if debug_mode:
            for line in output_scop_error.decode('utf-8').split('\n'):
                print('DEBUG:', line, '\n')
        raise CalledProcessError(1, 'pocc --output-scop')
    else:
        if debug_mode:
            print('DEBUG: output_scop_output')
            for line in output_scop_output.decode('utf-8').split('\n'):
                print('DEBUG:', line)
            print('\n\n\n')  '''

    # Obtainment of code
    code = get_code_from_scoplib('polypy.pocc.c.scop', numpy_format)

    # Cleanup
    if not debug_mode:
        if os.path.exists('polypy.scop'):
            os.remove('polypy.scop')
        if os.path.exists('polypy.pocc.c'):
            os.remove('polypy.pocc.c')
        if os.path.exists('polypy.pocc.c.scop'):
            os.remove('polypy.pocc.c.scop')
        if os.path.exists('polypy.pocc.pocc.c'):
            os.remove('polypy.pocc.pocc.c')
        if os.path.exists('polypy.pocc.pocc.c.scop'):
            os.remove('polypy.pocc.pocc.c.scop')

    return code


def __decompile_header(function):
    """
        Creates the code definition of a function based on a function bytecode

    Parameters
    ----------
    function : the function to get the header

    Returns
    -------
        A code string of the definition of the function provided
    """
    code = 'def '
    code += str(function.__code__.co_name)
    code += '('
    code += ','.join(function.__code__.co_varnames[:function.__code__.co_argcount])
    code += '):\n'
    return code



def __decompile_code(code, limit):
    """
        Generates a string code of a list of instructions. It is use to generate the code before the scop
    Parameters
    ----------
        code : list of bytecode instructions
        limit : index (integer) that indicates where to stop generating code 


    Returns
    -------
        The string of code corresponding to the selected list of instructions 
    """
    res = ''
    body = []
    self = 0
    for i in range(limit):
        line = code[i]
        if line.opname == 'STORE_FAST':
            b = str(body.pop())
            res+='    ' + line.argval + '=' + b + '\n'
        elif line.opname == 'STORE_SUBSCR':
            b = str(body.pop())  # Index
            b1 = str(body.pop())  # Varname
            b2 = str(body.pop())
            res+='    ' + b1 + '[' + b + ']=' + b2 + '\n'       
        elif line.opname == 'LOAD_FAST':
            if line.argval != "self":
                body.append(format(line.argval))
            else:
                self = 1
        elif line.opname == 'LOAD_CONST' or line.opname == "LOAD_ATTR":
            if self:
                s = 'self.' + format(line.argval)
                self = 0
            else:
                s = format(line.argval)                    
            body.append(s)
        elif line.opname == 'LOAD_GLOBAL':    
            body.append(line.argval)          
        elif line.opname == 'BINARY_ADD' or line.opname == 'INPLACE_ADD':
            b = body.pop()
            b1 = body.pop()
            body.append('(' + b1 + '+' + b + ')')
        elif line.opname == 'BINARY_SUBTRACT' or line.opname == 'INPLACE_SUBTRACT':
            b = str(body.pop())
            b1 = str(body.pop())
            body.append('(' + b1 + '+(-1*' + b + '))')
        elif line.opname == 'BINARY_MULTIPLY' or line.opname == 'INPLACE_MULTIPLY':
            b = str(body.pop())
            b1 = str(body.pop())
            body.append('(' + b1 + '*' + b + ')')
        elif line.opname == 'BINARY_TRUE_DIVIDE' or line.opname == 'INPLACE_TRUE_DIVIDE':
            b = str(body.pop())
            b1 = str(body.pop())
            body.append('(' + b1 + '/' + b + ')')
        elif line.opname == 'BINARY_SUBSCR':
            b = body.pop()
            b1 = body.pop()
            body.append( b1 + '[' + b + ']')
        elif line.opname == 'UNARY_NEGATIVE':
            b = str(body.pop())
            body.append('(-1*' + b + ')')     
        elif line.opname == 'LOAD_METHOD':
            if self:
                b = 'self'
                self = 0
            else:
                b = body.pop()
            body.append(b + '.' + line.argval)
        elif line.opname == 'BUILD_LIST':
            b = []
            for _ in range(line.argval):
                b.append(body.pop())
            b.reverse()
            body.append( '[' + ', '.join(b) + ']')
        elif line.opname == 'CALL_FUNCTION' or line.opname == 'CALL_METHOD':
            b = []
            for _ in range(line.argval):
                b.append(body.pop())
            b.reverse()
            body.append(str(body.pop()) + '(' + ', '.join(b) + ')')
        elif line.opname == 'EXTENDED_ARG':
            pass
        else:
            print("DEBUG: argument not considered", line.opname)
            for x in code:
                print("DEBUG: code block: ", x)
            raise ValueError('DEBUG: value not considered')
    return res


def __decompile_code_before_scop(code):
    """
        Searchs if exists code before de start of the scop. If true calls __decompile_code to generate that code
    Parameters
    ----------
        code : list of bytecode instructions

    Returns
    -------
        The string of code that represents the code before a scop, if it exists
    """
    start = 0
    loop_start = 0
    res = ''
    for i, line in enumerate(code):
        if line.opname == 'FOR_ITER':
            start = i
            break
    if start != 0:    
        loop_start = __find_loop_header_start(code, start)    
    
    if loop_start != 0:
        res = __decompile_code(code, loop_start)
       
    return res


def __decompile_return(return_bytecode):
    """
        Creates the return statement based on the bytecode of the return part
    Parameters
    ----------
        return_bytecode : list of instructions that make the return statement

    Returns
    -------
        The string of code corresponding to the return statement
    """
    code = ''
    stack = []

    code += __decompile_code(return_bytecode, len(return_bytecode)-1)
    for line in return_bytecode:
        if line.opname == 'LOAD_FAST':
            stack.append(line.argval)
        if line.opname == 'LOAD_CONST':
            stack.append(line.argval)
        elif line.opname == 'LOAD_NAME':
            stack.append(line.argval)
        elif line.opname == 'LOAD_GLOBAL':
            stack.append(line.argval)
        elif line.opname == 'BUILD_TUPLE':
            args = []
            for _ in range(line.argval):
                args.append(str(stack.pop()))
            args.reverse()
            stack.append('(' + ','.join(args) + ')')
        elif line.opname == 'RETURN_VALUE':
            return_value = stack.pop()
            if return_value is not None:
                code += '    return ' + str(return_value)
            else:
                return code + ''

    return code


def __generate_scoplib_from_loop(code_block):
    """
        Generates a scoplib file from a given bytecode
    Parameters
    ----------
    code_block: list of bytecode instructions from a loop

    Returns
    -------
        A scoplib file
    """
    # Creation of SCoPLib object
    var_id_by_var_name = {}
    statements = []
    parameters = []
    number = [0]

    stack = 0
    start = 0
    base_level_scattering = -1
    for i, _ in enumerate(code_block):
#        if code_block[i].opname == 'SETUP_LOOP':
        if code_block[i].opname == "FOR_ITER":
            stack += 1
            if stack == 1:
                start = __find_loop_header_start( code_block, i )
        elif code_block[i].opname == 'JUMP_ABSOLUTE':
            stack -= 1
            if not stack:
                statements.extend(
                    __obtain_statements_info(code_block[start:i + 1], [], [], var_id_by_var_name, number, parameters,
                                             [['BaseLevelScattering', base_level_scattering]]))
                base_level_scattering += 1

    scoplib = ScopLib()
    # Necesario que sea C para PoCC
    scoplib.language = 'C'
    scoplib.context = [0, 2 + len(parameters)]
    scoplib.parameters = parameters
    scoplib.statements = statements

    return scoplib

def __find_loop_header_start( code, start ):
    """
        Finds where a loop start
    Parameters
    ----------
        code: list of bytecode instructions
        start: index in the list that shows where to start to find the loop start

    Returns
    -------
        An index (integer) where the loop start
    """

    args = []
    for x in range( start-1, -1, -1 ):
        if code[x].opname == "GET_ITER": continue
        if code[x].opname == "EXTENDED_ARG": continue
        if code[x].opname == "CALL_FUNCTION":
            args.append( code[x].arg )
            continue
        if (code[x].opname == "LOAD_ATTR") or (code[x].opname == "LOAD_FAST") or (code[x].opname == "LOAD_CONST"):
            if not args: raise ValueError
            if code[x].argval == "self": continue
            args[-1] -= 1
            continue
        if (code[x].opname == "BINARY_SUBTRACT") or (code[x].opname == "BINARY_ADD"):
            if not args: raise ValueError
            args[-1] += 1 # This is one arg, but has 2 args in turn
            continue
        if code[x].opname  == "LOAD_GLOBAL":
            if len(args)>1: 
                args.pop(-1)
                continue
            if args[0] == 0: return x
        raise ValueError

def __separate_return_block(code):
    """
        Separates the body of a function from the return statement
    Parameters
    ----------
        code: list of bytecode instructions of a function

    Returns
    -------
        A tuple in which its first value is the code corresponding to the body of a function and the second to
        the return value
    """
    first_setup_loop = None
    last_pop_block = None
    for i, line in enumerate(code):
#        if line.opname == 'SETUP_LOOP' and first_setup_loop is None:
        if line.opname == "FOR_ITER" and first_setup_loop is None:
            first_setup_loop = __find_loop_header_start( code, i )
        elif line.opname == 'JUMP_ABSOLUTE':
            last_pop_block = i

    return code[first_setup_loop:last_pop_block + 1], code[last_pop_block + 1:]


def generate_scoplib_file(function, filename=''):
    """
        Generates a .scop file from the given function
    Parameters
    ----------
        function: list of bytecode instructions of a loop
        filename: name of the resulting file (default = the same as the function)

    Returns
    -------
        None
    """
    if filename == '':
        filename = function.__code__.co_name
    filename += '.scop'

    code = list(get_instructions(function))
    function_body, _ = __separate_return_block(code)

    scoplib = __generate_scoplib_from_loop(function_body)

    with open(filename, 'w') as scoplib_file:
        scoplib_file.write(scoplib.file_representation())


def get_code_from_scoplib(scoplib_file, numpy_format):
    """
        Generates a string of code representing the information on a scoplib file
    Parameters
    ----------
        scoplib_file: a scoplib file
        numpy_format: if True, code with NumPy format will be generated

    Returns
    -------
        A string of code representing the scoplib object

    """
    scoplib = ScopLib(scoplib_file)
    tree = __tree_from_statements(scoplib.statements, scoplib.parameters)

    if numpy_format:
        __get_numpy_format(tree)
    code = __get_code(tree)
    return code


def optimize(function, numpy_format, debug_mode=False):
    """
        Returns a function optimized with the polyhedral model. If the function cannot be
            optimized the original function is returned

    Parameters
    ----------
        function : the function to be optimized
        numpy_format : if True the code will be generated with NumPy format 
        debug_mode : if True, the optimized function will be printed in the console and the
            optimization files will
        not be deleted. (default = False)

    Returns
    -------
        The function optimized
    """
    code = list(get_instructions(function))
    optimized_code = ''

    _, function_return = __separate_return_block(code)

    optimized_code += __decompile_header(function)
    optimized_code += __decompile_code_before_scop(code)
    try:
        optimized_code += __optimize_loops(function, numpy_format, debug_mode)
    except CalledProcessError:
        generate_scoplib_file(function, 'aux')
        optimized_code += get_code_from_scoplib('aux.scop', False)
        if os.path.exists('aux.scop'):
            os.remove('aux.scop')

    optimized_code += __decompile_return(function_return)

    if debug_mode:
        print(optimized_code)

    optimized_bytecode = compile(optimized_code, '', 'exec')

    return optimized_bytecode
