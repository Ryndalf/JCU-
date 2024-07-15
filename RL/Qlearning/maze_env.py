import numpy as np
import time
import tkinter as tk
from tkinter import ttk
import pandas as pd
from tqdm import tqdm
from RL_brain import QlearningTable

# Amazing Demo -> title

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        print('Initiate environment parameters...')
        for _ in tqdm(range(100)):
            time.sleep(0.01)
        print('\n')
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Amazing Demo')
        self.width = 760
        self.geometry(f'{self.width}x650')
        self.iconphoto(False, tk.PhotoImage(file='./LOGO.jpg'))
        # self.geometry('{0}x{1}'.format(self.MAZE_W * UNIT, self.MAZE_H * UNIT))
        self.hell_index = []
        self.min_path = 0
        self.count = 0
        self.current_path = 10000
        self.speed = 0
        self._build_frame()
        self.frame4 = tk.Frame(self)
        self.frame4.pack()
        self.frame5 = tk.Frame(self)
        self.frame5.pack()

    def _build_frame(self):
        print('Initiate basic components...')
        for _ in tqdm(range(100)):
            time.sleep(0.01)
        print('\n')

        label1 = tk.Label(self, text='Reinforcement Learning (Supporting automatic driving)', font=("Times", 25))
        label1.pack()

        strings = 'Reinforcement Learning (RL) is a machine learning approach inspired by ' \
                  'how humans and animals learn through trial and error. ' \
                  'It focuses on maximizing cumulative rewards by interacting ' \
                  'with an environment, making it well-suited for tasks where ' \
                  'decisions must be learned based on feedback.'
        label2 = tk.Label(self, text=strings, font=("Times, 12"), wraplength=self.width*0.9, anchor='w', justify='left')
        label2.pack()

        sep1 = ttk.Separator(self, orient=tk.HORIZONTAL)
        sep1.pack(fill=tk.X, padx=5, pady=10)

        frame1 = tk.Frame(self)
        frame1.pack()

        frame1_label = tk.Label(frame1, text='Speed')
        frame1_label.pack(side='left')

        self.selected_value1 = tk.StringVar()
        combobox1 = ttk.Combobox(frame1, textvariable=self.selected_value1)
        combobox1['values'] = ('Fast', 'Medium', 'Slow')  # 设置选项
        combobox1.pack(side='left')

        self.selected_value2 = tk.StringVar()
        frame1_label2 = tk.Label(frame1, text='Map size')
        frame1_label2.pack(side='left')
        combobox2 = ttk.Combobox(frame1, textvariable=self.selected_value2)
        combobox2['values'] = ('5', '7', '9')  # 设置选项
        combobox2.pack(side='left')

        self.selected_value3 = tk.StringVar()
        frame1_label3 = tk.Label(frame1, text='Treasure position')
        frame1_label3.pack(side='left')
        combobox2 = ttk.Combobox(frame1, textvariable=self.selected_value3)
        combobox2['values'] = ('center', 'edge')  # 设置选项
        combobox2.pack(side='left')

        frame2 = tk.Frame(self)
        frame2.pack()

        sep2 = ttk.Separator(frame2, orient=tk.HORIZONTAL)
        sep2.pack(fill=tk.X, padx=5, pady=10)

        button1 = tk.Button(frame2, text="START", command=self.start_command)
        button1.pack(side='left', padx=70)

        button2 = tk.Button(frame2, text="SPEED CHANGE", command=self.speed_change)
        button2.pack(side='left', padx=70)

        button3 = tk.Button(frame2, text="FINISH", command=self.finish_comman)
        button3.pack(side='left', padx=70)

    def _build_maze(self, number_of_map, pos):
        frame3 = tk.Frame(self.frame4)
        frame3.pack(side='left')
        self.canvas = tk.Canvas(frame3, bg='white', height=self.MAZE_H * self.UNIT, width=self.MAZE_W * self.UNIT)
        self.canvas.pack(side='left', anchor='sw')
        for c in range(0, self.MAZE_W * self.UNIT, self.UNIT):
            x0, y0, x1, y1 = c, 0, c, self.MAZE_H * self.UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.MAZE_H * self.UNIT, self.UNIT):
            x0, y0, x1, y1 = 0, r, self.MAZE_W * self.UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([self.UNIT / 2, self.UNIT / 2])
        # hell_list = []
        color_size = self.UNIT * 0.8 / 2

        map_load = pd.read_csv(f'./map/{number_of_map}_{pos}.csv', index_col=0)
        for i in range(map_load.shape[0]):
            row = map_load.iloc[i, 0]
            col = map_load.iloc[i, 1]
            # hell_list.append((row, col))
            hell_center = origin + np.array([self.UNIT * row, self.UNIT * col])
            hell = self.canvas.create_rectangle(
                hell_center[0]-color_size, hell_center[1]-color_size,
                hell_center[0]+color_size, hell_center[1]+color_size,
                fill='black')
            self.hell_index.append(self.canvas.coords(hell))
        # hell_list = pd.DataFrame(hell_list, columns=['row', 'col'])
        # hell_list.to_csv(f'./map/{self.MAZE_H}_{pos}.csv')
        if pos == 'center':
            oval_center = origin + int(self.MAZE_H/2) * self.UNIT
        else:
            oval_center = origin + self.UNIT * (self.MAZE_H - 1)

        self.oval = self.canvas.create_oval(
            oval_center[0]-color_size, oval_center[1]-color_size,
            oval_center[1]+color_size, oval_center[1]+color_size,
            fill='yellow')

        self.rect = self.canvas.create_rectangle(
            origin[0]-color_size, origin[0]-color_size,
            origin[1]+color_size, origin[1]+color_size,
            fill='red')
        self.canvas.pack(side='left')

        frame5_1 = tk.Frame(self.frame5)
        frame5_1.pack(side='left')
        canvas1_frame5_1 = tk.Canvas(frame5_1, bg='yellow', width=40, height=40)
        canvas1_frame5_1.pack(side='left')
        label1_frame5_1 = tk.Label(frame5_1, text='Treasure', font=("Times", 12),
                                   width=10, height=1, anchor="w")
        label1_frame5_1.pack(side='left')
        frame5_2 = tk.Frame(self.frame5)
        frame5_2.pack(side='left')
        canvas_frame5_2 = tk.Canvas(frame5_2, bg='red', width=40, height=40)
        canvas_frame5_2.pack(side='left')
        label_frame5_2 = tk.Label(frame5_2, text='Robot', font=("Times", 12),
                                  width=10, height=1, anchor="w")
        label_frame5_2.pack(side='left')

        frame5_3 = tk.Frame(self.frame5)
        frame5_3.pack(side='left')
        canvas_frame5_3 = tk.Canvas(frame5_3, bg='black', width=40, height=40)
        canvas_frame5_3.pack(side='left')
        label_frame5_3 = tk.Label(frame5_3, text='Block', font=("Times", 12),
                                  width=10, height=1,anchor="w")
        label_frame5_3.pack(side='left')

        frame5_4 = tk.Frame(self.frame5)
        frame5_4.pack(side='left')
        canvas_frame5_4 = tk.Canvas(frame5_4, bg='white', width=40, height=40)
        canvas_frame5_4.create_rectangle(2, 2, 38, 38,outline="black", fill="white")
        canvas_frame5_4.pack(side='left')
        label_frame5_4 = tk.Label(frame5_4, text='Path', font=("Times", 12),
                                  width=10, height=1,anchor="w")
        label_frame5_4.pack(side='left')

        self.tree = ttk.Treeview(self.frame4, columns=['feature', 'up', 'down', 'right', 'left'],
                                 show='headings')
        self.tree.heading("feature", text="feature")
        self.tree.column("feature", width=70)
        self.tree.heading("up", text="up")
        self.tree.column("up", width=70)
        self.tree.heading("down", text="down")
        self.tree.column("down", width=70)
        self.tree.heading("right", text="right")
        self.tree.column("right", width=70)
        self.tree.heading("left", text="left")
        self.tree.column("left", width=70)
        self.tree.configure(height=3)
        self.tree.pack(fill='x', pady=0)

        label_path = f"Length of optimal path:{self.min_path}  " \
                     f"Length of current path:{self.current_path}"
        self.label_str2_ = tk.Label(self.frame4, text=label_path, font=("Times", 12),
                              wraplength=int(self.width*0.95/ 2), anchor='w', justify='left')
        self.label_str2_.pack()
        sep1 = ttk.Separator(self.frame4, orient=tk.HORIZONTAL)
        sep1.pack(fill=tk.X, padx=1, pady=5)
        label_str = 'Reward setting: \n' \
                    '    Reward is positive if robot move to treasure; \n' \
                    '    Reward is negative if robot move to block; \n'\
                    '    Reward is zero if robot move to path;\n' \
                    '******************************************\n' \
                    'Observation: \n' \
                    '    Next state obtained after executing the action in the header at the current position.\n' \
                    '******************************************\n' \
                    'Action:\n' \
                    '    Robot will choose the action with highest reward with a 90% probability marked as Bingo.' \
                    'Otherwise,choose randomly.\n'
        label_str_ = tk.Label(self.frame4, text=label_str, font=("Times", 12),
                              wraplength=int(self.width*0.95/2), anchor='w', justify='left')
        label_str_.pack()

    def reset(self):
        color_size = self.UNIT * 0.8 / 2
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([self.UNIT / 2, self.UNIT / 2])
        self.rect = self.canvas.create_rectangle(
            origin[0]-color_size, origin[1]-color_size,
            origin[0]+color_size, origin[1]+color_size,
            fill='red')
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0: # up
            if s[1] > self.UNIT:
                base_action[1] -= self.UNIT
        elif action == 1:  # down
            if s[1] < (self.MAZE_H - 1)* self.UNIT:
                base_action[1] += self.UNIT
        elif action == 2:  # right
            if s[0] < (self.MAZE_W - 1)* self.UNIT:
                base_action[0] += self.UNIT
        elif action == 3:  # left
            if s[0] > self.UNIT:
                base_action[0] -= self.UNIT
        self.canvas.move(self.rect, base_action[0], base_action[1])
        time.sleep(self.speed)
        self.count += 1
        s_ = self.canvas.coords(self.rect)
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in self.hell_index:
            reward = -1
            done = False
        else:
            reward = 0
            done = False
        return s_, reward, done

    def update_info(self, q_value, flag, action):
        q_value = q_value.to_list()
        q_value.insert(0, 'Reward')
        total_list = []
        total_list.append(q_value)
        feature_list = ['Observation']
        for i in q_value:
            if isinstance(i, str):
                continue
            if i >= 0:
                feature_list.append('path')
            else:
                feature_list.append('block')
        total_list.append(feature_list)
        q_value_table = pd.DataFrame(data=total_list, columns=['FEATURE', 'UP', 'DOWN', 'RIGHT', 'LEFT'], )
        self.tree.delete(*self.tree.get_children())
        for index, row in q_value_table.iterrows():
            self.tree.insert("", "end", values=row.tolist())
        if flag == True:
            values = ['Action', '', '', '', '']
            values[action+1] = 'Random'
            self.tree.insert("", "end", values=values)
        else:
            values = ['Action', '', '', '', '']
            values[action+1] = 'Bingo'
            self.tree.insert("", "end", values=values)

    def render(self):
        time.sleep(0.1)
        self.update()

    def get_opt_path(self, number_of_map, pos):
        if number_of_map == 5 and pos == 'center':
            self.min_path = 4
        elif number_of_map == 5 and pos == 'edge':
            self.min_path = 8
        elif number_of_map == 7 and pos == 'center':
            self.min_path = 6
        elif number_of_map == 7 and pos == 'edge':
            self.min_path = 12
        elif number_of_map == 9 and pos == 'center':
            self.min_path = 10
        elif number_of_map == 9 and pos == 'edge':
            self.min_path = 16
        else:
            self.min_path = 0

    def get_speed(self, speed):
        if speed == 'Fast':
            self.speed = 0
        elif speed == "Medium":
            self.speed = 0.25
        else:
            self.speed = 1

    def start_command(self):
        for widget in self.frame4.winfo_children():
            widget.destroy()
        for widget in self.frame5.winfo_children():
            widget.destroy()
        speed = self.selected_value1.get()
        self.get_speed(speed)
        number_of_map = int(self.selected_value2.get())
        pos = self.selected_value3.get()
        self.MAZE_W = number_of_map
        self.MAZE_H = number_of_map

        self.get_opt_path(number_of_map, pos)

        if number_of_map == 5:
            self.UNIT = 75
        elif number_of_map == 7:
            self.UNIT = 55
        else:
            self.UNIT = 45
        self._build_maze(number_of_map, pos=pos)
        RL = QlearningTable(actions=list(range(env.n_actions)))
        self.start_flag = True
        for episode in tqdm(range(100)):
            observation = env.reset()
            self.current_path = 0
            while self.start_flag:
                env.render()
                action, q_value, rand_flag = RL.choose_action(str(observation))
                env.update_info(q_value, rand_flag, action)
                observation_, reward, done = env.step(action)
                RL.learn(str(observation), action, reward, str(observation_))
                observation = observation_
                if done:
                    self.current_path = self.count
                    label_path = f"Length of optimal path:{self.min_path}\t" \
                                 f"Length of current path:{self.current_path}"
                    self.label_str2_.config(text=label_path)
                    self.count = 0
                    print(RL.q_table)
                    break

    def finish_comman(self):
        self.count = 0
        self.start_flag = False
        self.update()

    def speed_change(self):
        speed = self.selected_value1.get()
        self.get_speed(speed)
        self.update()

if __name__ == '__main__':
    env = Maze()
    env.mainloop()


