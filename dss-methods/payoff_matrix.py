import numpy as np


def compute_payoff_matrix(params_weights_dict, criteria_weights):
    '''
    Функция расчета платежной матрицы

            Parameters:
                    params_weights_dict: словарь с матрицами весов альтернатив для разных внешних состояний из метода аналитических иерархий
                    criteria_weights: массив с весами критериев

            Returns:
                    payoff_matrix: платежная матрица
    '''
    payoff_matrix = []
    for param, weight_matrix in params_weights_dict.items():
        payoff_matrix.append(weight_matrix @ criteria_weights.T)
    payoff_matrix = np.array(payoff_matrix).T
    return payoff_matrix

def laplace(matrix):
    '''
    Критерий Лапласа. Находит максимум средних по строкам и возвращает индекс соответствующей строки.

            Parameters:
                    matrix: платежная матрица с коэффициентами

            Returns:
                    best_path: индекс строки, в которой максимальное среднее
    '''
    best_path = np.argmax(matrix.mean(axis=1))
    return best_path + 1

def wald(matrix):
    '''
    Критерий Вальда. Находит максимум минимумов по строкам и возвращает индекс соответствующей строки.

            Parameters:
                    matrix: платежная матрица с коэффициентами

            Returns:
                    best_path: индекс строки, в которой максимальный минимум
    '''
    best_path = np.argmax(matrix.min(axis=1))
    return best_path + 1

def optimist(matrix):
    '''
    Критерий максимакса или оптимизма. Находит максимум максимумов по строкам и возвращает индекс соответствующей строки.

            Parameters:
                    matrix: платежная матрица с коэффициентами

            Returns:
                    best_path: индекс строки, в которой максимальный минимум
    '''
    best_path = np.argmax(matrix.max(axis=1))
    return best_path + 1

def pessimist(matrix):
    '''
    Критерий пессимизма. Находит минимум минимумов по строкам и возвращает индекс соответствующей строки.

            Parameters:
                    matrix: платежная матрица с коэффициентами

            Returns:
                    best_path: индекс выбранной строки
    '''
    best_path = np.argmin(matrix.min(axis=1))
    return best_path + 1

def hurwitz(matrix, alpha=0.5):
    '''
    Критерий Гурвица. Находит максимум минимумов по строкам и возвращает индекс соответствующей строки.

            Parameters:
                    matrix: платежная матрица с коэффициентами
                    alpha: Коэффициент α принимает значения от 0 до 1. Если α стремится к 1, то критерий Гурвица приближается к критерию максимакса,
                                а при α стремящемуся к 0, то критерий Гурвица приближается к критерию Вальда. По умолчанию равен 0.5


            Returns:
                    best_path: индекс строки, в которой максимальный минимум
    '''
    maxs = matrix.max(axis=1)
    mins = matrix.min(axis=1)
    best_path = np.argmax(alpha*maxs + (1-alpha)*mins)
    return best_path + 1

def savage(matrix):
    '''
    Критерий Сэвиджа или минимакса (критерий потерь). Строит матрицу потерь, в матрице потерь находит минимум максимумов по строкам и возвращает индекс соответствующей строки.

            Parameters:
                    matrix: платежная матрица с коэффициентами

            Returns:
                    best_path: индекс строки, в которой максимальный минимум
    '''
    loss_matrix = matrix.max(axis=0) - matrix
    best_path = np.argmin(loss_matrix.max(axis=1))
    return best_path + 1

def multiplication(matrix):
    '''
    Критерий произведений. Находит максимум произведений по строкам и возвращает индекс соответствующей строки.

            Parameters:
                    matrix: платежная матрица с коэффициентами

            Returns:
                    best_path: индекс выбранной строки
    '''
    best_path = np.argmax(np.prod(matrix, axis=1))
    return best_path + 1

def bayes(matrix, probs): ### Можно использовать, если есть веса внешних состояний
    '''
    Критерий Байеса или среднего выигрыша. Находит максимум взвешенных сумм коэффициентов и возвращает индекс соответствующей строки.

            Parameters:
                    matrix: платежная матрица с коэффициентами
                    probs: массив с весами внешних состояний

            Returns:
                    best_path: индекс выбранной строки
    '''
    best_path = np.argmax((matrix * probs).sum(axis=1))
    return best_path + 1

def germeier(matrix, probs): ### Можно использовать, если есть веса внешних состояний
    '''
    Критерий Гермейера.

            Parameters:
                    matrix: платежная матрица с коэффициентами
                    probs: массив с весами внешних состояний

            Returns:
                    best_path: индекс выбранной строки
    '''
    best_path = np.argmax((matrix * probs).min(axis=1))
    return best_path + 1

def hodge_lehmann(matrix, probs, alpha=0.5): ### Можно использовать, если есть веса внешних состояний
    '''
    Критерий Ходжа-Лемана.

            Parameters:
                    matrix: платежная матрица с коэффициентами
                    probs: массив с весами внешних состояний
                    alpha: Коэффициент α принимает значения от 0 до 1. Если α стремится к 1, то критерий Ходжа-Лемана приближается к критерию Байеса,
                                а при α стремящемуся к 0, то критерий Ходжа-Лемана приближается к критерию Вальда. По умолчанию равен 0.5

            Returns:
                    best_path: индекс выбранной строки
    '''

    best_path = np.argmax(alpha * (matrix * probs).sum(axis=1) + (1-alpha) * matrix.min(axis=1))
    return best_path + 1

def germeier_hurwitz(matrix, probs, alpha=0.5): ### Можно использовать, если есть веса внешних состояний
    '''
    Критерий Гермейера-Гурвица.

            Parameters:
                    matrix: платежная матрица с коэффициентами
                    probs: массив с весами внешних состояний
                    alpha: Коэффициент α принимает значения от 0 до 1. По умолчанию равен 0.5

            Returns:
                    best_path: индекс выбранной строки
    '''

    matrix_with_probs = matrix * probs
    best_path = np.argmax(alpha * matrix_with_probs.max(axis=1) + (1-alpha) * matrix_with_probs.min(axis=1))
    return best_path + 1
