# -*- coding: utf-8 -*-
"""
Module containing the class Tree
"""
from statement import Statement

class Tree:
    """
    Class used to represent a simple tree

    Attributes
    ----------
        name : str
            The iterator of the loop represented by the tree if the tree represents a loop. None if
            the tree represents a Statement
        data: Statement
            Statement associated to the tree. None if the tree represents a Loop.
        scat: String
            String that represent the scattering indicated for this tree
        diff: Boolean
            True if the tree represents a loop that have statements childs with differents conditions
        branches : list
            List of Trees that are under the current tree's scope and share this tree conditions.
        conditions : list
            List of conditions shared among the trees that are under this.
        invert : list
            List that is use when the tree represents a statement to generate code properly
    """
    def __init__(self):
        """
        Instantiates a new Tree with the default values:
            name = None
            data = None
            branches = empty list
            conditions= empty list

        Returns
        -------
            A new tree with the default values
        """
        self.name = None
        self.data = None
        self.scat = '0'
        self.diff = False
        self.branches = []
        self.conditions = []
        self.invert = [0, 0]

    def as_str(self, indentation=1):
        """
        Returns a string that represents the tree in a verbose way for debug purposes.
        The indentation argument is used to add tabs at the beginning, useful when used by objects
        that contain variables as arguments.
        Parameters:
        ----------
            indentation : the number of tabs at the beginning of the returned string (default = 0)

        Returns
        -------
            A string representing the tree
        """
        string = str(self.name) + ' scat='+ self.scat + '\n'
        for condition in self.conditions:
            string += (condition.as_str(indentation + 1) + '\n')
        for branch in self.branches:
            string += '    ' * indentation + '└─' + str(branch.as_str(indentation + 1))
        if not self.branches:
            if type(self.data) is Statement:
                for loop in self.data.loops:
                    for c in loop.loop_conditions:
                        string += (c.as_str(indentation + 1) + '\n')
        return string

    def __repr__(self):
        """
        Returns a representation of the loop for use with the print function. Calls the as_str
        function with the default values.

        Returns
        -------
            A string representing the tree
        """
        return self.as_str()
