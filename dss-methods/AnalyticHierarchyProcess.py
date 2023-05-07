import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import numpy as np
import time
import pandas as pd


def total(matrix, num_of_params):
    '''общая функция суммирует столбец матрицы и возвращает 1d-массив

    Аргументы:
    матрица: это матрица попарного сравнения после ввода данных пользователем.
    num_of_params: количество факторов, взятых для сравнения'''
    tot = np.full((num_of_params), 0, dtype=float)
    for i in range(num_of_params):
        for j in range(num_of_params):
            tot[i] = tot[i] + matrix[j, i]
    return(tot)



def normalization(sum_of_column, matrix, num_of_params):
    ''''функция нормализации вычисляет матрицу с выводом из общей функции и возвращает матрицу
    
    Аргументы:
    sum_of_column: это сумма каждого столбца матрицы попарного сравнения, а также результат общей функции.
    матрица: это матрица попарного сравнения после ввода данных пользователем.
    num_of_params: количество факторов, взятых для сравнения'''
    norm = np.full((num_of_params, num_of_params), 1, dtype=float)
    for i in range(num_of_params):
        for j in range(num_of_params):
            norm[i, j] = matrix[j, i]/sum_of_column[i]
    norm_t = norm.transpose()
    return (norm_t)


def weight(normalized_matrix, num_of_params):
    '''функция веса вычисляет вес каждого фактора

    Аргументы:
    normalized_matrix: это матрица из функции нормализации, которая имеет нормализованное значение, вычисленное из суммы столбцов
    num_of_params: Количество факторов, взятых для сравнения'''
    li = []
    for i in range(num_of_params):
        wt = np.sum(normalized_matrix[[i]])/num_of_params
        li.append(round(wt, 3))
    return(li)

def isfloat(value): 
    """Функция, определяющая, какое значение важности введено"""
    
    try:
        float(value)
        return True
    except ValueError:
        return False


def vector(matrix_a):
    """Функция расчета вектора приоритета"""
    
    vect = matrix_a[:, :]
    w = vect / vect.sum()
        
    return w


def matrix_a(number):
    """Функция создания парной матрицы"""
    
    A = np.ones([number, number])
    for i in range(0, number):
        for j in range(0, number): #равен {}".format(self.params_alt[i], value))
            if i < j: # Все, что выше главной диагонали, меняем на введенные польз-лем значения
                a = str(input(f'Насколько лучше альтернатива {i+1} чем альтернатива {j+1}? Введите целое число от 0.1 до 9: '))
                print('')
                if isfloat(a) and (0.11 <= float(a) <= 9): 
                    A[i,j] = float(a)
                    A[j, i] = 1 / float(a)
                else: # Учитываем ошибки
                    print("Введены неверные данные, давайте попробуем ещё раз \n")
                    a = str(input(f'Насколько лучше альтернатива {i+1} чем альтернатива {j+1}? Введите целое число от 0.1 до 9: '))
                    print('')
                    if isfloat(a) and (0.11 <= float(a) <= 9): 
                        A[i,j] = float(a)
                        A[j, i] = 1 / float(a)
                    else:
                        print("Введены неверные данные. Альтернативы одинаково важны.")
                        a = 1
                        A[i, j] = float(a)
                        A[j, i] = 1 / float(a)
                          
    return A


class Criteria_calculator(): #для матриц критериев
    
    def __init__(self):

        self.params = [] #для сбора параметров вводимых пользователем
        self.input_params = widgets.Text(layout={'width': '1200px'}) #виджет текстового поля
        self.save_params_button = widgets.Button(description="Готово") #кнопска для сохранения параметров
        self.calculate_button = widgets.Button(description="Рассчитать") #кнопка для начала расчёта параметров
        self.descLabel = widgets.Label('Введите все факторы через запятую:') #виджет ввода параметров 
        self.grid = None
        self.inputs_widgets = {}
        self.output = widgets.Output()
        self.bottomWidgets = widgets.VBox() #виджеты для кнопок
        self.allwidgets = widgets.VBox([])

        self.save_params_button.on_click(self.params_save) #функция сохранения по кнопке
        self.calculate_button.on_click(self.on_calculate) #функция расчёта по клику кнопки
        self.calculate_button.add_class('calculate_button') #класс для расчёта по кнопкам
        
        #стиль поля ввода и кнопки расчёта
        display(HTML("""<style>
        .params_label { background: rgba(0,0,0,0.1); text-align:center;}
        .calculate_button { color: magenta; margin-top:20px;}
        </style>"""))
        
        global k_count #количество критериев
        global kriter #названия критериев
        
        global list_k #список весов критериев
        list_k = []
      
    
    #функция сохранения по кнопке   
    def params_save(self, change):
        
        self.params = self.input_params.value.split(',')
        self.grid = widgets.GridspecLayout(
            len(self.params)+1, len(self.params)+1)
        self.allwidgets.children = [
            self.descLabel,
            widgets.HBox([
                self.input_params, self.save_params_button]), self.grid, self.calculate_button, self.bottomWidgets, self.output]
        self.build_grid()
        with self.output:
            clear_output()
            print('Диагональный элемент должен быть равен 1')
            print('Значение верхней треугольной матрицы должно быть обратным соответствующей нижней треугольной матрице')
   
    #функция расчёта по клику кнопки         
    def on_calculate(self, change): 
        
        user_input_matrix = self.convertInputToNumpyarray()
        column_sums = total(
            matrix=user_input_matrix, num_of_params=len(self.params))
        normalized_matrix = normalization(
            sum_of_column=column_sums, matrix=user_input_matrix, num_of_params=len(self.params))
        wts = weight(normalized_matrix=normalized_matrix,
                     num_of_params=len(self.params))

        with self.output:
            clear_output()
            print('Расчет...')
            time.sleep(1)
            clear_output()
            for i, value in enumerate(wts):
                print("Вес для '{}' равен {}".format(self.params[i], value))
                global list_k
                list_k.append(value) #наполнение списка весами критериев
            print('')

            global kriter
            kriter = self.params
    
            global k_count
            k_count = len(self.params)
            
            
    #для расстановки приоритетов
    def create_Input(self, default=0):

        return widgets.Dropdown(
            options=[('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8), ('9', 9),
                     ('1/2', 1/2), ('1/3', 1/3), ('1/4', 1/4), ('1/5', 1/5), ('1/6', 1/6), ('1/7', 1/7), ('1/8', 1/8), ('1/9', 1/9)],
            value=default,
            layout=widgets.Layout(width='50px',),
            disabled=False,
        )

    
    def get_input_widget(self, owner):
        
        for key, value in self.inputs_widgets.items():
            if(value == owner):
                k = key
                break
        return k

    #функция конвертирует полученные значения в матрицу
    def convertInputToNumpyarray(self):
        
        length = len(self.params)
        mat = np.full((length, length), 1, dtype=float)
        for i in range(1, length+1):
            for j in range(1, length+1):
                mat[i-1, j-1] = self.grid[i, j].value
        return mat

    
    def onchange(self, change):
        
        owner = change['owner']
        [row, column] = self.get_input_widget(owner).split('-')
        row = int(row)
        column = int(column)
        if(row != column and (row != 0 and column != 0)):
            self.grid[column, row].value = 1/change['new']
        if(row == column and (row != 0 and column != 0)):
            self.grid[column, row].value = 1
        

    def build_grid(self):
        
        for i in range(len(self.params)+1):
            for j in range(len(self.params)+1):
                if(i != j and (i == 0 or j == 0)):
                    if(i == 0):
                        labelindex = j-1
                    else:
                        labelindex = i-1

                    self.grid[i, j] = widgets.Label(
                        self.params[labelindex],

                    )
                    self.grid[i,j].add_class('params_label')

                if(i != 0 and j != 0):
                    self.grid[i, j] = self.create_Input(
                        default=2 if i > j else 1/2)

                    self.inputs_widgets['{}-{}'.format(i, j)] = self.grid[i, j]
                    self.grid[i, j].observe(self.onchange, 'value')

                if(i == j and (i != 0 and j != 0)):
                    self.grid[i, j] = self.create_Input(default=1,)
                    self.inputs_widgets['{}-{}'.format(i, j)] = self.grid[i, j]
                    self.grid[i, j].observe(self.onchange, 'value')
       
    #функция открытия и запуска всех виджетов для расчётов              
    def open_calc(self):

        self.allwidgets.children = [
            self.descLabel,
            widgets.HBox([
                self.input_params, self.save_params_button]), self.bottomWidgets, self.output]

        return self.allwidgets
    

Calc=Criteria_calculator() 
Calc.open_calc()


def main():
    global number
    number = '3' #Количество альтернатив
    
    if number.isdigit(): 
        A = matrix_a(int(number))
        weights = vector(A)
        
        for i in range(int(number)):
            print(f'Вес альтернативы {i+1} = {np.round(weights[i].sum(), 3)}')
    
            alt = np.round(weights[i].sum(), 3) 
            list_m.append(alt) #наполнение списка весами альтернатив
           

    else:
        print("Что-то пошло не так. Попробуйте еще раз. \n")
        main() 
        
    
list_fm = []
vns = (np.array(input("Введите внешние состояния через запятую: ").split(',')))
for j in range(len(vns)):   
    print()
    print("\033[1;32mРассмотрим внешнее состояние - {}, и произведём рассчёты ниже.\033[0m".format(vns[j]))
    print()
    list_m = [] #список весов альтернатив  
    for i in range(k_count):
        print()
        print("\033[1;34mРасчитаем для критерия «{}»:\033[0m".format(kriter[i]))
        print()

        if __name__ == "__main__":
            main()

    array_m = np.array(list_m) #массив весов альтернатив 
    array_m.resize(k_count, int(number))

    array_k = np.array(list_k) #массив весов критериев, которые можно использовать дальше    

    final_matrix = np.round(array_m.transpose(), 3)
    list_fm.append(final_matrix) #массив всех конечных матриц для метода анализа иерархий
    print()
    print("\033[1;35mМатрица весов альтернатив: \033[0m", + final_matrix, sep='\n\n')
          
