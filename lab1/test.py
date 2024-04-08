import io
import sys
import unittest
from pyknow import *
from version1 import Algoritmth, Fact

class TestAlgoritmth(unittest.TestCase):
    def test_sorting_merge(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('Сортировка'), Fact('нет информации о данных'), Fact('нужна стабильность')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('Сортировка слиянием', output_value)

    def test_sorting_quick(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('Сортировка'),Fact('нет информации о данных')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('Быстрая сортировка', output_value)

    def test_sorting_count(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('задача упорядочивания'),Fact('количество данных больше диапазона, в котором они заключены')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('Сортировка подсчетом', output_value)

    def test_sorting_karman(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('Сортировка'), Fact('равномерное распределение данных')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('Карманная сортировка', output_value)

    def test_sorting_radix(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('упорядочивание текста'),Fact('задача связана с текстом'), Fact('Алфавит входных сильно меньше кол-ва входных данных')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('Поразрядная сортировка', output_value)

    def test_text_lzw(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('Уменьшить размер текста'), Fact('задача связана с текстом'), Fact('неизвестные входные данные'), Fact('необходимо кодир повторяющиеся шаблоны данных')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('LZW', output_value)

    def test_text_haffman(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('Уменьшить размер текста'), Fact('задача связана с текстом'), Fact('неизвестные входные данные'), Fact('необходимо кодир отдельные символы')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('Алгоритмы Хаффмана', output_value)

    def test_text_rle(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('Сжатие текста'), Fact('Входные данные - длинная посл-ть повторяющихся символов')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('RLE', output_value)

    def test_text_kmp(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('Поиск в тексте'), Fact('задача связана с текстом'), Fact('префикс-функция'), Fact('Поиск подстрок'), Fact('текст похож на естественный язык')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('КМП', output_value)

    def test_text_z_func(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('Поиск в тексте'), Fact('задача связана с текстом'), Fact('простая реализация')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('z-функция', output_value)

    def test_trie_axo_korasik(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('Деревья'), Fact('Поиск подстрок'), Fact('Подстроки всегда одинаковые')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('Ахо-Корасик', output_value)

    def test_trie_patricia(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('Trie'), Fact('текст где ищем подстроки всегда один и тот же')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('PATRICIA', output_value)
    def test_balanced_tree_decart(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('сбалансированные деревья'), Fact('можно пожертвовать скоростью ради удобства написания кода')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('декартово дерево', output_value)
    def test_balanced_tree_rb(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('сбалансированные деревья'),Fact('требуется использовать больше операций вставки и удаления чем поиска')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('КЧ дерево/AVL дерево', output_value)
    def test_trie_aho(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('Trie'), Fact('Подстроки всегда одинаковые')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('Ахо-Корасик', output_value)
    def test_graph_bug1(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('нахождение кратчайшего расстояния в графе'),Fact('кратчайший путь между всеми парами вершин')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('алгоритм флойда-уоршалла', output_value)
    def test_graph_bug2(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('нахождение кратчайшего расстояния в графе'),Fact('кратчайший путь между всеми парами вершин')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('алгоритм флойда-уоршалла', output_value)
    def test_graph_b_tree(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('сбалансированные деревья'),Fact('необходима эффективная работа с дисковой памятью')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('B дерево', output_value)
    def test_graph_ostovnoe_derevo(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('Деревья'),Fact('необходимо определить наиболее экономичный способ соединения всех узлов')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('поиск минимального остовного дерева', output_value)
    def test_optimize_dynamic(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('задачи на оптимизацию'),Fact('потребуется ли сохранить результаты предыдущих этапов')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('динамическое программирование', output_value)
    def test_optimize_greedy(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('задачи на оптимизацию'),Fact('выбор лучшего из решений подзадач приводит к решению всей задачи')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('жадный алгоритм', output_value)
    def test_optimize_dijkstra(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
        Fact('нахождение кратчайшего расстояния в графе'),Fact('вес ребер должен быть неотрицательным')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('алгоритм дейкстры', output_value)
    def test_long(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
       Fact('графы'),Fact('найти кратчайшее расстояние между элементами'),Fact('вес ребер должен быть неотрицательным')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('алгоритм дейкстры', output_value)
    def test_long1(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
       Fact('в задаче используется структура данных, предназначенная для хранения информации'),Fact('состоит из вершин'),Fact('состоит из рёбер'),Fact('найти кратчайшее расстояние между элементами'),Fact('вес ребер должен быть неотрицательным')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('алгоритм дейкстры', output_value)
    def test_long2(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
       Fact('графы'),Fact('все вершины достижимы'), Fact('граф без циклов'), Fact('необходимо иметь минимальное возможное время доступа к элементам'),Fact('можно пожертвовать скоростью ради удобства написания кода')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('декартово дерево', output_value)
    def test_long3(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expert_system = Algoritmth()
        expert_system.reset()
        expert_system.factz([
       Fact('графы'),Fact('все вершины достижимы'), Fact('граф без циклов'), Fact('необходимо иметь минимальное возможное время доступа к элементам'),Fact('требуется использовать больше операций вставки и удаления чем поиска')])
        expert_system.run()
        output_value = captured_output.getvalue()
        sys.stdout = sys.__stdout__
        self.assertIn('КЧ дерево/AVL дерево', output_value)

if __name__ == '__main__':
    unittest.main()
