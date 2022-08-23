import inspect
from copy import deepcopy
from random import choice, choices, randint
from turtle import right
from typing import Callable, List, Tuple
import sympy as sym
from typing_extensions import Self


class Individual:
    def __init__(
        self, value: sym.Symbol | Callable | float
    ) -> None:
        self.value = value

        self.arity = len(inspect.signature(value).parameters) if callable(value) else 0
        self.left:  Individual = None
        self.right: Individual = None

    def copy(self):
        return deepcopy(self)

    def choose_random_node(self):
        tnode = choices(["leaf", "internal"], weights=[0.1, 0.9], k=1)[0]
        relevants = list(filter(lambda x: not callable(x[1]) if tnode == "leaf" else callable(x[1]), enumerate(self.get_function_arr())))
        return choice(relevants) if relevants else (0, self.value)


    def mutate(self, mutation_max_depth: int, functions: List[Callable], terminals: List):
        mutation_point = self.choose_random_node()

        target, t_depth = self.search(mutation_point[0])

        def mutate_callback(ind: Self):
            size = randint(0, mutation_max_depth - t_depth) if mutation_max_depth - t_depth > 0 else 0
            method = choice(["full", "grow"])
            new = Individual.random(size, method, functions, terminals)

            ind.arity = new.arity
            ind.value = new.value
            ind.left = new.left
            ind.right = new.right

        mutate_callback(target)




    def search(self, node: int, idx=0, depth=0) -> Tuple[Self, int]:
        if node == idx:
            return (self, depth)

        if self.left is not None:
            idx = self.left.search(node, idx+1, depth+1)
        else:
            return idx

        if isinstance(idx, tuple):
            return idx

        if self.right is not None:
            idx = self.right.search(node, idx+1, depth+1)

        return idx


    def get_depth(self, depth=0) -> int:
        leftd = 0
        rightd = 0
        if self.left is not None:
            leftd = self.left.get_depth(depth + 1)
        if self.right is not None:
            rightd = self.right.get_depth(depth + 1)

        if self.right is None and self.left is None:
            return depth

        return max((rightd, leftd, depth))

    def prune(self, target_depth: int, terminals: list, depth=0) -> None:
        if depth == target_depth:
            terminal = choice(terminals)
            if callable(terminal):
                terminal = terminal()
            self.value = terminal
            self.arity = 0
            self.left = None
            self.right = None

        self.left.prune(target_depth, terminals, depth + 1) if self.left is not None else None
        self.right.prune(target_depth, terminals, depth + 1) if self.right is not None else None



    def crossover(self, partner: Self, max_depth: int, terminals: list) -> Tuple[Self, Self]:
        p_copy = partner.copy()
        s_copy = self.copy()

        s_random = s_copy.choose_random_node()
        s_subtree, _ = s_copy.search(s_random[0])

        p_random = p_copy.choose_random_node()
        p_subtree, _ = p_copy.search(p_random[0])

        aux = s_subtree.copy()

        s_subtree.value = p_subtree.value
        s_subtree.arity = p_subtree.arity
        s_subtree.left = p_subtree.left
        s_subtree.right = p_subtree.right

        p_subtree.value = aux.value
        p_subtree.arity = aux.arity
        p_subtree.left =  aux.left
        p_subtree.right = aux.right

        if s_copy.get_depth() > max_depth:
            s_copy.prune(7, terminals)

        if p_copy.get_depth() > max_depth:
            p_copy.prune(7, terminals)

        return p_copy, s_copy

    @staticmethod
    def random(depth, method: str, functions: List[Callable], terminals: List[sym.Symbol], d=0):
        match method:
            case "full":
                if d < depth:
                    individual = Individual(choice(functions))
                else:
                    chosen = choice(terminals)
                    individual = Individual(choice(terminals))
                    if callable(chosen):
                        chosen = chosen()
                    individual = Individual(chosen)
                    return individual
                individual.left = Individual.random(depth, method, functions, terminals, d + 1)
                if individual.arity == 2:
                    individual.right = Individual.random(depth, method, functions, terminals, d + 1)
                return individual

            case "grow":
                if d < depth:
                    chosen = choice(choice([functions, terminals]))
                    if chosen in terminals:
                        if callable(chosen):
                            chosen = chosen()
                        return Individual(chosen)
                    individual = Individual(chosen)

                else:
                    chosen = choice(terminals)
                    individual = Individual(choice(terminals))
                    if callable(chosen):
                        chosen = chosen()
                    individual = Individual(chosen)
                    return individual

                individual.left = Individual.random(depth, method, functions, terminals, d + 1)
                if individual.arity == 2:
                    individual.right = Individual.random(depth, method, functions, terminals, d + 1)

                return individual
            case _:
                return Individual(0)

    def get_function_arr(self) -> List:
        left_func = self.left.get_function_arr() if self.left is not None else []
        right_func = self.right.get_function_arr() if self.right is not None else []
        return [self.value, *left_func, *right_func]

    def get_func(self):
        match self.arity:
            case 2:
                return self.value(self.left.get_func(), self.right.get_func())
            case 1:
                return self.value(self.left.get_func())
            case _:
                return self.value
