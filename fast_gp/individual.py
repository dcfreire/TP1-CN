import inspect
from ast import Expression
from copy import deepcopy
from random import choice, choices, randint
from typing import Callable, List, Tuple

import sympy as sym
from typing_extensions import Self


class Individual:
    def __init__(self, value: Callable | float | str) -> None:
        """
        Args:
            value (Callable | float | str): Valor do nodo, pode ser uma função, uma constante
            ou uma variável (string).
        """
        self.value = value

        self.arity = len(inspect.signature(value).parameters) if callable(value) else 0
        self.parent: Individual | None = None
        self.left: Individual | None = None
        self.right: Individual | None = None
        self.depth = 0

    def copy(self) -> Self:
        """Cria uma cópia do objeto e objetos apontados por ele.

        Returns:
            Individual: A cópia.
        """
        return deepcopy(self)

    def choose_random_node(self) -> Tuple[int, Callable | float | str]:
        """Escolhe um nodo aleatório da árvore, com 90% de chance de escolher um nodo interno
        e 10% de escolher uma folha.

        Returns:
            Tuple[int, Callable | float | str]: Tupla do índice do node e seu valor.
        """
        tnode = choices(["leaf", "internal"], weights=[0.1, 0.9], k=1)[0]
        relevants = list(
            filter(
                lambda x: not callable(x[1]) if tnode == "leaf" else callable(x[1]),
                enumerate(self.get_function_arr()),
            )
        )
        return choice(relevants) if relevants else (0, self.value)

    def eval(self, params: List[float]) -> float:
        """Avalia a árvore substituindo as variáveis pelas especificadas em params.

        Args:
            params (List[float]): Lista de parâmetros, onde o índice é o nome do parâmetro.

        Returns:
            float: Função avaliada.
        """
        if callable(self.value):
            if self.arity == 2:
                return self.value(self.left.eval(params), self.right.eval(params))
            elif self.arity == 1:
                return self.value(self.left.eval(params))
        return self.value if isinstance(self.value, (float, int)) else params[int(self.value)]

    def mutate(self, mutation_max_depth: int, functions: List[Callable], terminals: List):
        """Realiza uma mutação em si mesmo.

        Args:
            mutation_max_depth (int): Limite da profundidade da árvore após a mutação.
            functions (List[Callable]): Lista de funções disponíveis para realizar a mutação.
            terminals (List): Lista de terminais disponíveis para realizar a mutação.
        """
        mutation_point = self.choose_random_node()

        target, t_depth = self.search(mutation_point[0])

        size = randint(0, mutation_max_depth - t_depth) if mutation_max_depth - t_depth > 0 else 0
        method = choice(["full", "grow"])
        new = Individual.random(size, method, functions, terminals)

        target.arity = new.arity
        target.value = new.value
        target.left = new.left
        target.right = new.right
        target.depth = new.depth

        self.fix_tree(None)

        target.update_parent_depth()

    def search(self, node: int, __idx=0, __depth=0) -> Tuple[Self, int]:
        """Busca na árvore o nodo no índice node.

        Args:
            node (int): Índice do nodo.
            idx (int, optional): Índice atual. Defaults to 0.
            depth (int, optional): Profundidade atual. Defaults to 0.

        Returns:
            Tuple[Self, int]: O indivíduo e sua profundidade na árvore.
        """
        if node == __idx:
            return (self, __depth)

        if self.left is not None:
            __idx = self.left.search(node, __idx + 1, __depth + 1)
        else:
            return __idx

        if isinstance(__idx, tuple):
            return __idx

        if self.right is not None:
            __idx = self.right.search(node, __idx + 1, __depth + 1)

        return __idx

    def get_depth(self, __depth=0) -> int:
        """Calcula a profundidade da árvore.

        Args:
            __depth (int, optional): Profundidade atual. Defaults to 0.

        Returns:
            int: Profundidade da árvore.
        """
        leftd = 0
        rightd = 0
        if self.left is not None:
            leftd = self.left.get_depth(__depth + 1)
        if self.right is not None:
            rightd = self.right.get_depth(__depth + 1)

        if self.right is None and self.left is None:
            return __depth

        return max((rightd, leftd, __depth))

    def prune(self, target_depth: int, terminals: list, __depth=0) -> None:
        """Poda a árvore por baixo, substituindo as partes podadas por terminais.

        Args:
            target_depth (int): Profundidade alvo da poda.
            terminals (list): Lista de terminais disponíveis.
            depth (int, optional): Profundidade atual. Defaults to 0.
        """
        if __depth == target_depth:
            terminal = choice(terminals)
            if callable(terminal):
                terminal = terminal()
            self.value = terminal
            self.arity = 0
            self.left = None
            self.right = None

        self.left.prune(target_depth, terminals, __depth + 1) if self.left is not None else None
        self.right.prune(target_depth, terminals, __depth + 1) if self.right is not None else None

    def prune_top(self, target_depth: int) -> None:
        """Poda a árvore por cima, escolhendo aleatoriamente um
        dos filhos e fazendo dele a nova raiz.

        Args:
            target_depth (int): Profundidade alvo da poda.
        """
        while self.depth > target_depth:
            if self.left is not None and self.right is not None:
                lower = choice([self.left, self.right])
            else:
                lower = self.left

            self.value = lower.value
            self.arity = lower.arity
            self.right = lower.right
            self.left = lower.left
            self.depth = lower.depth

    def update_parent_depth(self):
        """Atualiza a profundidade dos pais depois de uma cruzamento, ou mutação."""
        if self.parent is not None:
            if self.parent.right is not None:
                if self.parent.right.depth > self.depth:
                    self.parent.right.update_parent_depth()
                    return
                if self.parent.left.depth > self.depth:
                    self.parent.left.update_parent_depth()
                    return

            self.parent.depth = self.depth + 1
            self.parent.update_parent_depth()

        return

    def fix_tree(self, parent: Self | None):
        """Concerta as relações de pai e filho na árvore. Essa função foi implementada para
        concertar um erro de origem não determinada, onde alguns nodos de repente tinham a variável
        self.parent setada para None ou para o pai errado.

        Args:
            parent (Self): Pai do nodo atual.
        """
        self.parent = parent
        if self.right is not None:
            self.right.fix_tree(self)
        if self.left is not None:
            self.left.fix_tree(self)

    def crossover(self, partner: Self, max_depth: int) -> Tuple[Self, Self]:
        """Realiza o crossover entre self e partner.

        Args:
            partner (Self): Com quem será realizado o crossover.
            max_depth (int): Profundidade máxima dos filhos.

        Returns:
            Tuple[Self, Self]: Tupla com os dois filhos gerados.
        """
        p_copy = partner.copy()
        s_copy = self.copy()

        s_random = s_copy.choose_random_node()
        s_subtree, s_depth = s_copy.search(s_random[0])

        p_random = p_copy.choose_random_node()
        p_subtree, p_depth = p_copy.search(p_random[0])

        if p_depth + s_subtree.depth > max_depth:
            s_subtree.prune_top(max_depth - p_depth)

        if s_depth + p_subtree.depth > max_depth:
            p_subtree.prune_top(max_depth - s_depth)

        aux = s_subtree.copy()

        s_subtree.value = p_subtree.value
        s_subtree.arity = p_subtree.arity
        s_subtree.left = p_subtree.left
        s_subtree.right = p_subtree.right
        s_subtree.depth = p_subtree.depth

        p_subtree.value = aux.value
        p_subtree.arity = aux.arity
        p_subtree.left = aux.left
        p_subtree.right = aux.right
        p_subtree.depth = aux.depth

        s_copy.fix_tree(None)
        p_copy.fix_tree(None)

        p_subtree.update_parent_depth()
        s_subtree.update_parent_depth()

        return p_copy, s_copy

    @staticmethod
    def random(
        depth: int, method: str, functions: List[Callable], terminals: List[float | str], __d=0, __parent=None
    ) -> "Individual":
        """Cria um individuo aleatório com profundidade depth, utilizando o método especificado,
        as funções e terminais.

        Args:
            depth (int): Profundidade alvo
            method (str): Método a utilizar, podendo ser "grow", "full". Caso nenhum
            dos dois seja passado retorna um indivíduo com valor 0.
            functions (List[Callable]): Funções para serem usadas na criação do indivíduo.
            terminals (List[float  |  str]): Terminais para serem usadas na criação do indivíduo.
            d (int, optional): Profundidade atual. Defaults to 0.
            parent (Individual, optional): Pai do indivíduo atual. Defaults to None.

        Returns:
            Individual: Indivíduo aleatório.
        """
        match method:
            case "full":
                if __d < depth:
                    individual = Individual(choice(functions))
                else:
                    chosen = choice(terminals)
                    if callable(chosen):
                        chosen = chosen()
                    individual = Individual(chosen)
                    individual.parent = __parent

                    return individual
                individual.left = Individual.random(depth, method, functions, terminals, __d + 1, individual)
                if individual.arity == 2:
                    individual.right = Individual.random(
                        depth, method, functions, terminals, __d + 1, individual
                    )
                individual.depth = individual.get_depth()
                individual.parent = __parent

                return individual

            case "grow":
                if __d < depth:
                    chosen = choice(choice([functions, terminals]))
                    if chosen in terminals:
                        if callable(chosen):
                            chosen = chosen()
                        individual = Individual(chosen)
                        individual.parent = __parent

                        return individual
                    individual = Individual(chosen)

                else:
                    chosen = choice(terminals)
                    individual = Individual(choice(terminals))
                    if callable(chosen):
                        chosen = chosen()
                    individual = Individual(chosen)
                    individual.parent = __parent

                    return individual

                individual.left = Individual.random(depth, method, functions, terminals, __d + 1, individual)
                if individual.arity == 2:
                    individual.right = Individual.random(
                        depth, method, functions, terminals, __d + 1, individual
                    )
                individual.depth = individual.get_depth()
                individual.parent = __parent

                return individual
            case _:
                return Individual(0)

    def get_function_arr(self) -> List:
        """Retorna uma lista que representa a árvore

        Returns:
            List: Lista que representa a árvore
        """
        left_func = self.left.get_function_arr() if self.left is not None else []
        right_func = self.right.get_function_arr() if self.right is not None else []
        return [self.value, *left_func, *right_func]

    def get_func(self):
        """Retorna a função simbólica da árvore

        Returns:
            Expression: Função simbólica
        """
        match self.arity:
            case 2:
                return self.value(self.left.get_func(), self.right.get_func())
            case 1:
                return (
                    sym.cos(self.left.get_func())
                    if self.value.__name__ == "cos"
                    else sym.sin(self.left.get_func())
                )
            case _:
                return sym.Symbol(f"{chr(97 + int(self.value))}") if isinstance(self.value, str) else self.value
