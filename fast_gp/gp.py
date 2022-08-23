from random import choices
from typing import Callable, List, Tuple

import numpy as np

from .functions import add, cos, mul, rand_f, sin, sub
from .individual import Individual


def protected_division(x, y):
    if y == 0:
        return 1
    return x / y


class GP:
    """Classe de programação genética. Após sua inicialização você pode iniciar o
    algoritmo com GP(*args).start().
    """

    def __init__(
        self,
        data: np.ndarray,
        popsize: int,
        mutation_rate: float,
        min_depth: int,
        max_depth: int,
        tournament_size: int,
        generations: int,
        phi: float,
        n_elites: int,
        elitist_operators=True,
    ) -> None:
        """
        Args:
            data (np.ndarray): Dados de treinamento.
            popsize (int): Tamanho da população.
            mutation_rate (float): Taxa de mutação.
            min_depth (int): Profundidade Mínima das soluções.
            max_depth (int): Profundidade Máxima das soluções.
            tournament_size (int): Número de indivíduos em cada torneio.
            generations (int): Número de gerações.
            phi (float): \"Learning rate\", quanto maior mais rápido as soluções
                         convergem. phi=0 desabilita essa funcionalidade.
            n_elites (int): Número de elites (distintos) para passar para próxima geração
            elitist_operators (bool, optional): True: Utilizar operadores elitistas,
                                                False: Não utilizar operadores elitistas.
                                                Defaults to True.
        """
        self.phi = phi
        self.functions: List[Callable] = [add, protected_division, sub, mul, sin, cos]
        self.terminals: List = [f"{i}" for i in range(len(data[0]) - 1)] + [rand_f]
        self.data = data
        self.generations = generations
        self.popsize = popsize
        self.pop: List[Tuple[float, Individual]] = []
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.divisor = np.sum((self.data[:, -1] - np.mean(self.data[:, -1])) ** 2)
        self.it = 0
        self.elitist_operators = elitist_operators
        self.n_elites = n_elites

    def _gen_pop(self):
        for i in range(self.popsize):
            ind = Individual.random(
                (i // (self.popsize / (self.max_depth - 1))) + self.min_depth,
                "grow" if i % 2 else "full",
                self.functions,
                self.terminals,
            )
            self.pop.append((self.fitness(ind), ind))

    def tournament(self, subpop: List[Tuple[float, Individual]]) -> Tuple[float, Individual]:
        """Retorna o melhor indivíduo de subpop.

        Args:
            subpop (List[Tuple[float, Individual]]): Lista de indivíduos e suas respectivas fitness.

        Returns:
            Tuple[float, Individual]: Tupla de fitness e indivíduo.
        """
        return min(subpop, key=self.get_fitness)

    def get_fitness(self, ind: Tuple[float, Individual] | Tuple[float, Individual, str]) -> float:
        """Retorna a fitness da tupla de indivíduo.

        Args:
            ind (Tuple[float, Individual] | Tuple[float, Individual, str]): Tupla fitness, indivíduo.

        Returns:
            float: Fitness.
        """
        return ind[0]

    def get_weighted_fitness(self, ind: Tuple[float, Individual] | Tuple[float, Individual, str]) -> float:
        """Retorna a fitness penalizada para repetições.

        Args:
            ind (Tuple[float, Individual] | Tuple[float, Individual, str]): Tupla fitness, individuo,
            e opcionalmente se o indivíduo foi pai ou filho na seleção atual.

        Returns:
            float: Fitness penalizada.
        """
        if not self.phi:
            return self.get_fitness(ind)
        progress = self.it / self.generations
        div_p = len([1 for i, _ in self.pop if i == ind[0]]) / self.popsize
        div = np.log(self.phi + div_p / progress) / np.log(self.phi)
        return ind[0] * div

    @staticmethod
    def get_result_from_future(future):
        return future.result()[:2]

    def get_elites(self) -> List[Tuple[float | Individual]]:
        """Retorna os self.n_elites melhores indivíduos.

        Returns:
            List[Tuple[float | Individual]]: Melhores indivíduos distintos.
        """
        sorted_pop = sorted(self.pop, key=self.get_fitness)
        ret = []
        r_fits = []
        for ind in sorted_pop:
            if ind[0] not in r_fits:
                ret.append(ind)
                r_fits.append(ind[0])
            if len(ret) == self.n_elites:
                break

        return ret

    def selection(self) -> Tuple[int, int]:
        """Realiza os torneios para seleção de pais para o cruzamento e do pai para mutação, conta
        o número de filhos de cruzamento e mutação superiores aos pais.

        Returns:
            Tuple[int, int]: Número de filhos de cruzamento superiores aos pais e
            filhos de mutação superiores aos pais respectivamente.
        """
        # with ProcessPoolExecutor(max_workers=1) as executor:
        offspring = []
        psize = self.popsize - self.n_elites
        while len(offspring) / psize < 1 - self.mutation_rate:
            offspring.append(self.crossover_worker())

        child_count = sum(map(lambda x: 1 if x[2] == "child" else 0, offspring))

        mut_count = 0
        while len(offspring) < psize:
            ind = self.mutation_worker()
            if ind[2] == "child":
                mut_count += 1
            offspring.append(ind)

        offspring.extend(self.get_elites())
        offspring = list(map(lambda x: x[:2], offspring))

        self.pop = offspring
        return child_count, mut_count

    def crossover_worker(self) -> Tuple[float, Individual, str]:
        """Realiza o crossover, se estivermos utilizando operadores elitistas retorna o melhor entre os filhos e pais,
        caso contrário retorna o melhor dos filhos.

        Returns:
            Tuple[float, Individual, str]: Tripla fitness, indivíduo e se é filho ou pai.
        """
        tourn1 = (*self.tournament(choices(self.pop, k=self.tournament_size)), "parent")
        tourn2 = (*self.tournament(choices(self.pop, k=self.tournament_size)), "parent")
        of1, of2 = tourn1[1].crossover(tourn2[1], self.max_depth)
        of1 = (self.fitness(of1), of1, "child")
        of2 = (self.fitness(of2), of2, "child")
        return (
            min([tourn1, tourn2, of1, of2], key=self.get_weighted_fitness)
            if self.elitist_operators
            else min([of1, of2], key=self.get_weighted_fitness)
        )

    def mutation_worker(self) -> Tuple[float, Individual, str]:
        """Realiza a mutação, se estivermos utilizando operadores elitistas retorna o melhor entre o filho e pai,
        caso contrário retorna o filho.

        Returns:
            Tuple[float, Individual, str]: Tripla fitness, indivíduo e se é filho ou pai.
        """
        old = (*self.tournament(choices(self.pop, k=self.tournament_size)), "parent")
        new = old[1].copy()
        new.mutate(self.max_depth, self.functions, self.terminals)
        new = (self.fitness(new), new, "child")
        return min([old, new], key=self.get_weighted_fitness) if self.elitist_operators else new

    def fitness(self, ind: Individual, data=None) -> float:
        """Calcula o NMRSE do individuo ind.

        Args:
            ind (Individual): Individuo para calcular a fitness.
            data (_type_, optional): Se devemos calcular sobre um conjunto de
            dados diferente de self.data. Defaults to None.

        Returns:
            float: Fitness (NMRSE) de ind.
        """
        total = 0
        data = self.data if data is None else data
        divisor = self.divisor if data is None else self.test_div

        for line in data:
            y = line[-1]
            ev = ind.eval(line[:-1])
            total += (y - ev) ** 2

        return np.sqrt(total / divisor)

    def start(self, test: np.ndarray | None = None) -> Tuple[Tuple[float, Individual], dict]:
        """Inicia o processo evolucionário

        Args:
            test (np.ndarray | None, optional): Conjunto de dados teste. Defaults to None.

        Returns:
            Tuple[Tuple[float, Individual], dict]: Tupla com uma tupla de fitness e individuo e um dicionário com o log
            do processo evolucionário.
        """
        if test is not None:
            self.test_div = np.sum((test[:, -1] - np.mean(test[:, -1])) ** 2)
        self._gen_pop()
        log = {
            "popsize": self.popsize,
            "mutation_rate": self.mutation_rate,
            "tournament_size": self.tournament_size,
            "generations": self.generations,
            "phi": self.phi,
            "n_elites": self.n_elites,
            "iteration": [],
        }
        better_childs, better_mut = (0, 0)
        while self.it < self.generations:
            if self.generations - self.it == 5:
                self.phi = 0
            best = min(self.pop, key=self.get_fitness)
            log["iteration"].append(
                {
                    "best": best[0],
                    "mean": np.mean(list(map(self.get_fitness, self.pop))),
                    "worst": max(map(self.get_fitness, self.pop)),
                    "diversity": len(set(map(self.get_fitness, self.pop))),
                    "better_child": better_childs,
                    "better_mut": better_mut,
                    "test": "None" if test is None else self.fitness(best[1], test),
                }
            )

            self.it += 1
            better_childs, better_mut = self.selection()
        return min(self.pop, key=self.get_fitness), log
