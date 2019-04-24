import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
import pylab
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


class Hexapod:
    def __init__(self, axis):
        """
        Инициализация начальных параметров системы
        :param axis: ось вращения системы (x, y, z)
        """
        self.axis = axis  # ось вращения тела
        self.alpha = 30.  # угол между приводами в точке крпления
        self.beta = 30.  # двугранный угол между приводами и платформой
        self.L = 1.5  # длина привода
        self.h_c = 2.  # высота центра масс тела
        self.r = 1.  # радиус тела
        self.m_p = 1000.  # масса платформы
        self.m = 4000.  # масса тела
        self.nu = 0.5  # частота

        # тензор инерции тела для решения обратной задачи
        self.J = np.array([[5000, 0, 0],
                           [0, 5000, 0],
                           [0, 0, 3500]], np.float32)

        # начальное положение точек крепления приводов на ВЕРХНЕЙ платформе
        self.A_0 = np.round([[self.r*np.sin(2*np.pi/3*i + np.pi),
                              self.r*np.cos(2*np.pi/3*i + np.pi),
                              -self.h_c] for i in range(-1, 2)], 5)

        # точки крепления приводов на НИЖНЕЙ платформе (const)
        self.B = np.array([])
        # положение точек крепления приводов на ВЕРХНЕЙ платформе за все время
        self.A = np.array([])
        # длина каждого привода за все время
        self.all_full_lengths = np.array([])
        # плечи сил приводов за все время
        self.r = np.array([])

        # ампилитуда вращения, закон изменения угла и его производные по OX
        self.fi_x_0 = 4.  # градусы
        self.fi_x = lambda t: self.fi_x_0 * np.sin(2*np.pi*self.nu*t)
        self.prime_fi_x = lambda t: self.fi_x_0 * 2*np.pi*self.nu * np.cos(2*np.pi*self.nu*t)
        self.prime2_fi_x = lambda t: -self.fi_x_0 * (2*np.pi*self.nu)**2 * np.sin(2*np.pi*self.nu*t)

        # ампилитуда вращения, закон изменения угла и его производные по OY и OZ
        self.fi_y_0 = 4.  # градусы
        self.fi_y = lambda t: self.fi_y_0 * np.sin(2*np.pi*self.nu*t)
        self.prime_fi_y = lambda t: self.fi_y_0 * 2*np.pi*self.nu * np.cos(2*np.pi*self.nu*t)
        self.prime2_fi_y = lambda t: -self.fi_y_0 * (2*np.pi*self.nu)**2 * np.sin(2*np.pi*self.nu*t)

        # матрица поворота вокруг оси OX
        self.R_matrix_x = lambda t: np.round([[np.cos(self.fi_x(t)*np.pi/180.), -np.sin(self.fi_x(t)*np.pi/180.), 0],
                                              [np.sin(self.fi_x(t)*np.pi/180.), np.cos(self.fi_x(t)*np.pi/180.), 0],
                                              [0, 0, 1]], 5)

        # матрица поворота вокруг оси OY
        self.R_matrix_y = lambda t: np.round([[1, 0, 0],
                                              [0, np.cos(self.fi_y(t)*np.pi/180.), -np.sin(self.fi_y(t)*np.pi/180.)],
                                              [0, np.sin(self.fi_y(t)*np.pi/180.), np.cos(self.fi_y(t)*np.pi/180.)]], 5)

        # матрица поворота вокруг оси OZ
        self.R_matrix_z = lambda t: np.round([[np.cos(self.fi_y(t)*np.pi/180.), 0, np.sin(self.fi_y(t)*np.pi/180.)],
                                              [0, 1, 0],
                                              [-np.sin(self.fi_y(t)*np.pi/180.), 0, np.cos(self.fi_y(t)*np.pi/180.)]], 5)

        # для построения геометрии точек B
        self.H = np.cos(np.pi/180. * self.beta) * np.cos(np.pi/180. * self.alpha/2) * self.L
        self.h = self.L * np.cos(np.pi/180.*self.alpha/2) * np.sin(np.pi/180.*self.beta)
        self.a = self.L * np.sin(np.pi/180.*self.alpha/2)  # основание треугольника
        self.r = (self.h**2 + self.a**2)**0.5

        # отсчет времени для расчета законов
        self.end_time = 2.0
        self.start_time = 0.
        self.steps = 100
        self.time = np.linspace(self.start_time, self.end_time, self.steps)

        # связь индексов нижней и верхней платформы
        self.indexes = [[0, 0], [0, 1], [1, 2], [1, 3], [2, 4], [2, 5]]

    def set_B(self):
        """
        Расчет геометрии стенда - положение точек B_i.

        Задается: self.B
        :return: None
        """
        for i, A in enumerate(self.A_0):
            a = A[:2]
            b1 = np.array([self.h, self.a])
            b2 = np.array([self.h, - self.a])

            kappa = np.array([[np.cos(np.pi / 180 * (30-120*i)), -np.sin(np.pi / 180 * (30-120*i))],
                              [np.sin(np.pi / 180 * (30-120*i)), np.cos(np.pi / 180 * (30-120*i))]])

            p1 = np.dot(kappa, b1) + a
            p2 = np.dot(kappa, b2) + a
            p1 = np.append(p1, - self.H - self.h_c)
            p2 = np.append(p2, - self.H - self.h_c)
            self.B = np.hstack((self.B, p1))
            self.B = np.hstack((self.B, p2))

        self.B = self.B.reshape(6, 3)

        # проверка длин приводов
        i = 0
        for A in self.A_0:
            assert np.linalg.norm(np.subtract(A, self.B[i])) - self.L <= 1e-4
            assert np.linalg.norm(np.subtract(A, self.B[i + 1])) - self.L <= 1e-4
            # print(np.linalg.norm(np.subtract(A, self.B[i])))
            # print(np.linalg.norm(np.subtract(A, self.B[i + 1])))
            i += 2

    def get_delta_L(self):
        """
        Расчет геометрии положения точек A_i в каждый момент времени.
        Отрисовка графиков изменения длин, скорости и ускорения для каждого привода по времени.

        Задается: self.A, self.all_full_lengths, self.set_r
        :return: None
        """
        print('####################################################')
        print('[INFO] solve delta L, Velocity, Acceleration ...')
        print('####################################################')
        # матрица поворота вокруг зазадной оси
        R_matrix = None
        if self.axis == 'x':
            R_matrix = self.R_matrix_x
        elif self.axis == 'y':
            R_matrix = self.R_matrix_y
        elif self.axis == 'z':
            R_matrix = self.R_matrix_z

        # удлинения каждого цилиндра за заданное время
        dL_all = []
        # длины всех цилиндров за все время
        L_all = []
        # координаты точек крепления на ВЕРХНЕЙ платформе
        coordinates_A = []
        # легенда для графиков
        colors = {0: 'r+--', 1: 'rx-',
                  2: 'g+--', 3: 'gx-',
                  4: 'b+--', 5: 'bx-'}

        for i, j in self.indexes:
            print('[INFO] Поршень №{}'.format(j+1))
            dl = []  # изменение длины поршня в момент времени t
            l = []  # длины поршня в момент времени t
            coord = []  # координата точки A_i в момент времени t
            for t in self.time:
                try:
                    A = np.dot(R_matrix(t), self.A_0[i])
                except Exception:
                    print('Type error axis')

                # текущая длина привода
                L = np.linalg.norm(self.B[j] - A)
                print(self.B[j] - A)
                print(self.L, L)
                # L = np.sum((A - self.B[j])**2)**0.5
                print('dL[мм] = {:.5f}'.format((L - self.L) * 1e3))
                l.append(L)
                dl.append(round(((L - self.L) * 1e3), 5))
                coord.append(A)

            dL_all.append(dl)
            L_all.append(l)
            coordinates_A.append(coord)

            # численно находим СКОРОСТЬ изменения длины приводов
            v = [0.0]
            for k in range(self.steps - 1):
                v.append((dl[k+1] - dl[k]) / (self.time[k+1] - self.time[k]))
            pylab.figure(1)
            pylab.plot(self.time[5:], v[5:], colors[j])
            print('[INFO] v_max =', np.max(np.abs(v[5:])))

            # численно находим УСКОРЕНИЕ изменения длины приводов
            a = [0.0]
            for k in range(self.steps - 1):
                a.append((v[k + 1] - v[k]) / (self.time[k + 1] - self.time[k]))
            pylab.figure(2)
            pylab.plot(self.time[5:], a[5:], colors[j])
            print('[INFO] a_max =', np.max(np.abs(a[5:])))
            print('****************************************************')

        # легенда для графика со скоростями
        pylab.figure(1)
        pylab.legend([r'1 line', '2 line', '3 line', '4 line', '5 line', '6 line'], loc=0)
        pylab.title('Velocity')
        pylab.xlabel('Time [s]')
        pylab.ylabel('Velocity [mm/s]')
        pylab.grid()
        # plt.savefig("output/velocity_{}.png".format(self.axis))

        # легенда для графика с ускорениями
        pylab.figure(2)
        pylab.legend([r'1 line', '2 line', '3 line', '4 line', '5 line', '6 line'], loc=0)
        pylab.title('Acceleration')
        pylab.xlabel('Time [s]')
        pylab.ylabel('Acceleration [mm/s^2]')
        pylab.grid()
        # pylab.savefig("output/acceleration_{}.png".format(self.axis))

        # график удлинения каждого поршня
        pylab.figure(3)
        for i in range(6):
            pylab.plot(self.time, dL_all[i], colors[i])
        pylab.legend([r'1 line', '2 line', '3 line', '4 line', '5 line', '6 line'], loc=0)
        pylab.title('Delta length')
        pylab.xlabel('Time [s]')
        pylab.ylabel('dL [mm]')
        pylab.grid()
        # pylab.savefig("output/length_{}.png".format(self.axis))
        plt.show()

        # исключим повторение вершин
        self.A = np.array(coordinates_A[0::2])
        self.all_full_lengths = np.array(L_all)
        self.set_r()
        # покадровая отрисовка геометрии стенда
        self.plot_3d_lines()

        # self.plot_animate(coordinates_A)

    def plot_3d_lines(self):
        """
        Покадровая отрисовка геометрии стенда в 3D.
        :return: None
        """
        pylab.figure(figsize=(12, 10))
        ax = pylab.axes(projection='3d')

        colors = {0: 'r', 1: 'orange',
                  2: 'g', 3: 'olive',
                  4: 'b', 5: 'navy'}
        markers = {0: '^', 1: '^',
                   2: 'o', 3: 'o',
                   4: '*', 5: '*'}

        # задать легенду
        for i, j in self.indexes:
            df_A = pd.Series(data=self.A_0[i], index=['x', 'y', 'z'])
            df_B = pd.Series(data=self.B[j], index=['x', 'y', 'z'])

            x = [df_A.x, df_B.x]
            y = [df_A.y, df_B.y]
            z = [df_A.z, df_B.z]
            ax.scatter(x, y, z, c=colors[j], marker=markers[j], s=20.)
        ax.legend([r'1', '2', '3', '4', '5', '6'], loc=0)

        # indexes = [[0, 0], [1, 2], [2, 4]]
        # построить смещения каждого поршня
        for i, j in self.indexes:
            k = 0
            for (a, r) in zip(self.A[i], self.r[j]):
                df_A = pd.Series(data=a, index=['x', 'y', 'z'])
                df_B = pd.Series(data=self.B[j], index=['x', 'y', 'z'])
                df_r = pd.Series(data=r, index=['x', 'y', 'z'])

                # геометрия длины цилиндров
                x = [df_A.x, df_B.x]
                y = [df_A.y, df_B.y]
                z = [df_A.z, df_B.z]

                # геометрия плеч сил
                x1 = [df_r.x, 0]
                y1 = [df_r.y, 0]
                z1 = [df_r.z, 0]

                # продолжение оси цилиндров
                x2 = [df_r.x, df_B.x]
                y2 = [df_r.y, df_B.y]
                z2 = [df_r.z, df_B.z]

                # частичная раскадровка
                if k % int(self.steps-1) == 0:
                # if k:
                    # ax.plot(x1, y1, z1, c=colors[j], marker=markers[j])
                    # ax.plot(x2, y2, z2, c='gray', marker='+')
                    ax.plot(x, y, z, c=colors[j], marker=markers[j])
                    # print('H_A =', z[0])
                k += 1

        # посторить смещение верхней плтаформы
        for i in range(0, self.steps, 9):
            a = np.array([self.A[0, i], self.A[1, i], self.A[2, i]])
            df_A = pd.DataFrame(data=a, columns=['x', 'y', 'z'])
            df_A = pd.concat((df_A, df_A.take([0])), axis=0)

            ax.plot(df_A.x.values, df_A.y.values, df_A.z.values, c='gray')

        # отрисовать начальные положения верхней и нижней платформы
        df_B = pd.DataFrame(data=self.B, columns=['x', 'y', 'z'])
        df_B = pd.concat((df_B, df_B.take([0])))
        df_A = pd.DataFrame(data=self.A_0, columns=['x', 'y', 'z'])
        df_A = pd.concat((df_A, df_A.take([0])), axis=0)

        ax.plot(df_B.x.values, df_B.y.values, df_B.z.values, c='black', linewidth=4.)
        ax.plot(df_A.x.values, df_A.y.values, df_A.z.values, c='black', linewidth=4.)

        ax.view_init(30, -39)
        # pylab.savefig("output/plot_3d_{}.png".format(self.axis))
        plt.show()

    def calculate_angles(self, l1, l2):
        """
        Решение теоремы косинусов для поиска угла по трем сторонам
        :param l1: прилежащая сторона к вычисляемому углу
        :param l2: противолежащая сторона к углу
        :return: (alpha, teta, gamma) - углы в треугольнике сил
        """
        cos_teta = (l1**2 + (2*self.a)**2 - l2**2) / 2*l1*2*self.a
        teta = np.arccos(cos_teta) * 180. / np.pi
        b = l1**2 + self.a**2 - 2*l1*self.a*cos_teta
        cos_alpha = (l1**2 + b**2 - l2**2) / 2*l1*self.a
        alpha = np.arccos(cos_alpha) * 180. / np.pi
        gamma = 180. - teta - alpha
        return alpha, teta, gamma

    def set_r(self):
        """
        Вычисление радиус-векторов плеч сил для каждого цилиндра
        :return: None
        """
        r_all = []
        for i, j in self.indexes:
            r = []
            for a in self.A[i]:
                L = np.array(a - self.B[j])
                direct_L = L / np.linalg.norm(L)
                t1 = np.array([0, -direct_L[2], direct_L[1]])
                b1 = np.array([a[2]*direct_L[1] - a[1]*direct_L[2]])
                t2 = direct_L
                b2 = np.array([0])
                t3 = np.array([a[1]*direct_L[2] - a[2]*direct_L[1],
                               -a[0]*direct_L[2] + a[2]*direct_L[0],
                               a[0]*direct_L[1] - a[1]*direct_L[0]])
                b3 = np.array([0])
                T = np.stack((t1, t2, t3))
                b = np.stack((b1, b2, b3))

                r.append(np.linalg.solve(T, b).reshape((3,)))
            r_all.append(r)

        self.r = np.array(r_all)

    def solve_dynamic_forces(self):
        """
        решение обратной задачи стенда
        Первое приближение  - решение двумерной зазадчи для пооврота оси вокруг оси х
        :return: минимальная и максимальная нагрузка на каждый цилиндр
        """
        print('####################################################')
        print('[INFO] solve DYNAMIC forces ...')
        print('####################################################')
        A = []
        for i in range(self.steps):
            a = []
            for j in range(3):
                a_ = self.A[j, i, :]
                a.append(a_)
            A.append(a)

        R = []
        for i in range(self.steps):
            r = []
            for j in range(6):
                r_ = self.r[j, i, :]
                r.append(r_)
            R.append(r)

        A = np.array(A)
        R = np.array(R)
        forces = []
        for a, r, t in zip(A, R, self.time):
            L = []
            direct = []  # направления сил
            shoulder = []  # плечи сил
            for i, j in self.indexes:
                len = np.array(self.B[j] - a[i])
                dir = len / np.linalg.norm(len)
                L.append(len)
                direct_force_try = self.B[j] - r[j]
                # direct.append(dir)
                direct.append(direct_force_try)
                # direct.append(direct_force_try)
                shoulder.append(np.cross(r[j], direct_force_try))
            L = np.array(L)

            T_static = np.array(direct).T
            T_dynamics = np.array(shoulder).T

            b_static = np.array([-self.m*9.8, 0, 0]).reshape((3, 1))

            # определение направления действующих сил
            dynamic_comp = None
            if self.axis == 'x':
                comp = self.J[2, 2] * self.prime2_fi_x(t)
                dynamic_comp = np.array([comp, 0, 0]).reshape((3, 1))
            elif self.axis == 'y':
                comp = self.J[1, 1] * self.prime2_fi_y(t)
                dynamic_comp = np.array([0, comp, 0]).reshape((3, 1))
            elif self.axis == 'z':
                comp = self.J[0, 0] * self.prime2_fi_y(t)
                dynamic_comp = np.array([0, 0, comp]).reshape((3, 1))
            b_dynamic = dynamic_comp

            # T = np.vstack((T_static, T_dynamics))
            # b = np.vstack((b_static, b_dynamic))

            T = T_dynamics[:, :3]
            b = b_dynamic[:, :3]
            # print(T)
            # print(b)

            dynamic_f = np.linalg.solve(T, b).reshape((3,))
            forces.append(dynamic_f)

            print('[INFO] time:', t)
            print('[INFO] length:', [round(np.linalg.norm(l), 4) for l in L])
            print('[INFO] shoulders:', [round(np.linalg.norm(l), 4) for l in np.array(shoulder)])
            print('[INFO] forces:', [round(f, 4) for f in dynamic_f])
            print('[INFO] dynamic component:', b_dynamic.T)
            print('****************************************************')
        forces = np.array(forces).T

        # график приложенной силы к цилиндрам от времени
        colors = {0: 'r+--', 1: 'rx-',
                  2: 'g+--', 3: 'gx-',
                  4: 'b+--', 5: 'bx-'}
        for i, j in self.indexes:
            pylab.plot(self.time, forces[i], colors[j])
        # colors = {0: 'r+--', 1: 'g+--', 2: 'b+--'}
        # for i in range(3):
        #     pylab.plot(self.time, forces[i], colors[i], label='$F_{}$'.format(i))
        pylab.legend([r'$F_1$', '$F_2$', '$F_3$', '$F_4$', '$F_5$', '$F_6$'], loc=0)
        # plt.legend(loc="lower right")
        pylab.title('Dynamic forces')
        pylab.xlabel('Time [s]')
        pylab.ylabel('Force [kg*m/s^2]')
        pylab.grid()

        plt.show()

    def solve_static_forces(self):
        """
        Решение обратной задачи стедна для статических нагрузок
        :return: компоненты силы для каждой опоры
        """
        print('####################################################')
        print('[INFO] solve STATIC forces ...')
        print('####################################################')
        x_symmetry_ind = [[0, 0], [0, 1], [1, 2]]
        forces = []
        for a, t in zip(self.A.reshape((self.steps, 3, 3)), self.time):
            L1 = np.array(a[0] - self.B[0])
            L2 = np.array(a[0] - self.B[1])
            L3 = np.array(a[1] - self.B[2])

            direct_L1 = L1 / np.linalg.norm(L1)
            direct_L2 = L2 / np.linalg.norm(L2)
            direct_L3 = L3 / np.linalg.norm(L3)

            T = np.stack((direct_L1, direct_L2, direct_L3))
            b = np.array([-self.m*9.8/2, 0, 0]).reshape((3, 1))

            static_f = np.linalg.solve(T, b).reshape((3,)) / 2
            forces.append(static_f)
            print('[INFO] time:', t)
            print('[INFO] length:',
                  round(np.linalg.norm(L1), 4),
                  round(np.linalg.norm(L2), 4),
                  round(np.linalg.norm(L3), 4))
            print('[INFO] forces:',
                  round(static_f[0]/2, 4),
                  round(static_f[1]/2, 4),
                  round(static_f[2]/2, 4))
            print('****************************************************')
        forces = np.array(forces).T

        # график приложенной силы к цилиндрам от времени
        colors = {0: 'r+--', 1: 'g+--', 2: 'b+--'}
        for i, j in x_symmetry_ind:
            pylab.plot(self.time, forces[j], colors[j])
        pylab.legend([r'$F_1$', '$F_2$', '$F_3$'], loc=0)
        pylab.title('Static forces')
        pylab.xlabel('Time [s]')
        pylab.ylabel('Force [kg*m/s^2]')
        pylab.grid()

        plt.show()

    def plot_animate(self, A):
        """"
        try to create animate function to plot mechanisms
        """
        fig = plt.figure()
        fig.set_tight_layout(False)
        ax = plt.axes(projection='3d')
        global cnt
        cnt = ax

        global cur_A
        global cur_B
        cur_A = A[0]
        cur_B = self.B[0]

        def steps(count=1):
            for i in range(count):
                df_A = pd.Series(data=cur_A[i], index=['x', 'y', 'z'])
                df_B = pd.Series(data=cur_B, index=['x', 'y', 'z'])
                x = [df_A.x, df_B.x]
                y = [df_A.y, df_B.y]
                z = [df_A.z, df_B.z]
                cnt.plot(x, y, z)

        def animate(frame):
            steps(1)
            return cnt
        anim = animation.FuncAnimation(fig, animate, frames=100)
        plt.show()


if __name__ == "__main__":
    hex = Hexapod(axis='y')

    hex.set_B()
    hex.get_delta_L()
    hex.solve_static_forces()
    hex.solve_dynamic_forces()