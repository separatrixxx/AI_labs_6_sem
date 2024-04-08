import sys
from pyknow import *

         
class Algoritmth(KnowledgeEngine):
    def __init__(self):
        super().__init__()

    @Rule(OR(
           AND(Fact('упорядочивание текста'),Fact('задача связана с текстом')),
           Fact('задача упорядочивания')))
    def sort(self):
        self.declare(Fact('Сортировка'))
        
    @Rule(AND(Fact('Сортировка'),Fact('нет информации о данных'), NOT(Fact('нужна стабильность'))))
    def quick(self):
        self.declare(Fact(algo='Быстрая сортировка'))
    
    @Rule(AND(Fact('Сортировка'), Fact('нет информации о данных'), Fact('нужна стабильность')))
    def merge(self):
        self.declare(Fact(algo='Сортировка слиянием'))
    
    @Rule(AND(Fact('Сортировка'), Fact('равномерное распределение данных')))
    def karman(self):
        self.declare(Fact(algo='Карманная сортировка'))
    
    @Rule(AND(Fact('Сортировка'), Fact('количество данных больше диапазона, в котором они заключены')))
    def count(self):
        self.declare(Fact(algo='Сортировка подсчетом'))
    
    @Rule(AND(Fact('Сортировка'), Fact('Алфавит входных сильно меньше кол-ва входных данных')))
    def radix(self):
        self.declare(Fact(algo='Поразрядная сортировка'))

    @Rule(AND(Fact('Уменьшить размер текста'), Fact('задача связана с текстом')))
    def text_zip(self):
        self.declare(Fact('Сжатие текста'))

    @Rule(AND(Fact('Сжатие текста'), Fact('неизвестные входные данные'), Fact('необходимо кодир повторяющиеся шаблоны данных')))
    def lzw(self):
        self.declare(Fact(algo='LZW'))

    @Rule(AND(Fact('Сжатие текста'), Fact('неизвестные входные данные'), Fact('необходимо кодир отдельные символы'))) # пересечение
    def haffman(self):
        self.declare(Fact(algo='Алгоритмы Хаффмана'))

    @Rule(AND(Fact('Сжатие текста'), Fact('Входные данные - длинная посл-ть повторяющихся символов')))
    def rle(self):
        self.declare(Fact(algo='RLE'))

    @Rule(AND(Fact('Поиск в тексте'), Fact('задача связана с текстом')))
    def substring(self):
        self.declare(Fact('Поиск подстрок'))

    @Rule(OR(AND(Fact('Поиск подстрок'), Fact('префикс-функция')), AND(Fact('Поиск подстрок'), Fact('текст похож на естественный язык'))))
    def kmp(self):
        self.declare(Fact(algo='КМП'))

    @Rule(AND(Fact('Поиск подстрок'), Fact('простая реализация')))
    def z_func(self):
        self.declare(Fact(algo='z-функция'))

    @Rule(AND(Fact('Деревья'), Fact('Поиск подстрок'))) #пересечение
    def trie(self):
        self.declare(Fact('Trie'))

    @Rule(AND(Fact('Trie'), Fact('Подстроки всегда одинаковые')))
    def axo_korasik(self):
        self.declare(Fact(algo='Ахо-Корасик'))

    @Rule(AND(Fact('Trie'), Fact('текст где ищем подстроки всегда один и тот же')))
    def patricia(self):
        self.declare(Fact(algo='PATRICIA'))
    
    @Rule(OR(
           AND(Fact('в задаче используется структура данных, предназначенная для хранения информации'),Fact('состоит из вершин'),Fact('состоит из рёбер')),
           Fact('задача связана с картами местности')))
    def graphs(self):
        self.declare(Fact('графы'))
        
    @Rule(AND(Fact('графы'),Fact('найти кратчайшее расстояние между элементами')))
    def graph_distance(self):
        self.declare(Fact('нахождение кратчайшего расстояния в графе'))

    @Rule(AND(Fact('нахождение кратчайшего расстояния в графе'),Fact('кратчайший путь между всеми парами вершин')))
    def floyd_warshall(self):
        self.declare(Fact(algo = 'алгоритм флойда-уоршалла'))
        
    @Rule(AND(Fact('нахождение кратчайшего расстояния в графе'),Fact('вес ребер должен быть неотрицательным')))
    def deikstra(self):
        self.declare(Fact(algo = 'алгоритм дейкстры'))
        
    @Rule(AND(Fact('графы'),OR(Fact('задача связана с поиском максимума'),Fact('задача связана с поиском минимума'))))
    def optimization(self):
        self.declare(Fact('задачи на оптимизацию'))
        
    @Rule(AND(Fact('задачи на оптимизацию'),Fact('выбор лучшего из решений подзадач приводит к решению всей задачи')))
    def jadniy(self):
        self.declare(Fact(algo='жадный алгоритм'))

    @Rule(AND(Fact('задачи на оптимизацию'),Fact('потребуется ли сохранить результаты предыдущих этапов')))
    def dp(self):
        self.declare(Fact(algo='динамическое программирование'))
        
    @Rule(AND(Fact('графы'),Fact('все вершины достижимы'), Fact('граф без циклов')))
    def tree(self):
        self.declare(Fact('Деревья'))
    
    @Rule(AND(Fact('Деревья'),Fact('необходимо определить наиболее экономичный способ соединения всех узлов')))
    def tree_min(self):
        self.declare(Fact(algo = 'поиск минимального остовного дерева'))
        
    @Rule(AND(Fact('Деревья'),Fact('необходимо иметь минимальное возможное время доступа к элементам')))
    def tree_balanced(self):
        self.declare(Fact('сбалансированные деревья'))
        
    @Rule(AND(Fact('сбалансированные деревья'),Fact('можно пожертвовать скоростью ради удобства написания кода')))
    def tree_decart(self):
        self.declare(Fact(algo ='декартово дерево'))
        
    @Rule(AND(Fact('сбалансированные деревья'),Fact('требуется использовать больше операций вставки и удаления чем поиска')))
    def tree_AVL(self):
        self.declare(Fact(algo ='КЧ дерево/AVL дерево'))
        
    @Rule(AND(Fact('сбалансированные деревья'),Fact('необходима эффективная работа с дисковой памятью')))
    def tree_B(self):
        self.declare(Fact(algo ='B дерево'))
    @Rule(Fact(algo=MATCH.a))
    def print_result(self,a):
          print('Алгоритм - {}'.format(a))
                    
    def factz(self,l):
        for x in l:
            self.declare(x)


if __name__ == "__main__":
    ex1 = Algoritmth()
    ex1.reset()
    ex1.factz([
        Fact('сбалансированные деревья'),Fact('можно пожертвовать скоростью ради удобства написания кода')])
    ex1.run()
