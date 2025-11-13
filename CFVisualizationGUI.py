import ast
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from PIL.ImageOps import expand
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from pandastable import Table
import os
import numpy as np
import matplotlib.colors as colors
#from sympy.plotting.pygletplot.util import create_bounds


class CSVGraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CF Visualization tool")

        self.file_path = None
        self.df = None
        self.env_type = None
        self.target = None

        self.create_widgets()

    def create_widgets(self):
        self.file_button = tk.Button(self.root, text="Select CSV file", command=self.load_csv)
        self.file_button.grid(row=0, column=0, columnspan=2, pady=10)

        self.frame_csv = tk.Frame(self.root, width=500, height=400)
        self.frame_csv.grid(row=1, column=0, sticky='nsew')

        self.canvas_csv = tk.Canvas(self.frame_csv)
        self.canvas_csv.pack(side='left', fill='both', expand=True)

        self.scrollbar_csv = ttk.Scrollbar(self.frame_csv, orient='vertical', command=self.canvas_csv.yview)
        self.scrollbar_csv.pack(side='right', fill='y')

        self.canvas_csv.configure(yscrollcommand=self.scrollbar_csv.set)
        self.canvas_csv.bind('<Configure>',
                             lambda e: self.canvas_csv.configure(scrollregion=self.canvas_csv.bbox("all")))

        self.frame_table = tk.Frame(self.canvas_csv)
        self.canvas_csv.create_window((0, 0), window=self.frame_table, anchor='nw')

        # Create a frame for the graph display
        self.frame_graph = tk.Frame(self.root, width=500, height=400)
        self.frame_graph.grid(row=1, column=1, sticky='nsew')

        self.title_label = tk.Label(self.frame_graph, text= '')
        self.title_label.pack()

        # Configure grid weights to make frames expandable
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)




    def load_csv(self):
        self.file_path = filedialog.askopenfilename(initialdir='C:/Users/MAHA164/RACCER-main/eval')
        if self.file_path:
            self.df = pd.read_csv(self.file_path)
            self.env_type = os.path.basename(os.path.dirname(os.path.dirname(self.file_path)))
            self.display_csv()

    def display_csv(self):
        self.table = Table(self.frame_table, dataframe = self.df)
        self.table.show()
        self.table.bind('<ButtonRelease-1>', self.on_click)

    def on_click(self, event):
        row_clicked = self.table.get_row_clicked(event)
        if row_clicked is not None:
            selected_data = self.df.iloc[row_clicked]
            self.generate_graph(selected_data, self.env_type)



    def generate_graph(self, row, env_type):
        for widget in self.frame_graph.winfo_children():
            if widget != self.title_label:
                widget.destroy()

        cf_list = row['cf']
        cf_list_object = ast.literal_eval(cf_list)
        fact_list = row['fact']
        fact_list_object = ast.literal_eval(fact_list)
        try:
            cf_path = self.preprocess_cf_path(row['cf_path'])
            cf_path_object = ast.literal_eval(cf_path)
        except SyntaxError as e:
            messagebox.showerror("Error", f"Invalid cf_path format: {e}")
            return


        cf_readable = row['cf_readable']
        fact_readable = row['fact_readable']
        self.target = row['target']

        combined_list = [fact_list_object[:-1]] + list(cf_path_object) + [cf_list_object[:-1]]

        keys = ['Fact']
        for i in range(len(cf_path_object)):
            keys.append(f'Step {i+1}')
        keys.append('Counterfactual')

        state_dictionary = dict(zip(keys, combined_list))




        if env_type=='frozen_lake':
            self.generate_graph_frozen_lake(state_dictionary)
        if env_type=='gridworld':
            self.generate_graph_gridworld(state_dictionary)

    def preprocess_cf_path(self, cf_path):
        # Remove any unwanted characters or brackets
        cf_path = re.sub(r'[()]', '', cf_path)
        return cf_path


    def generate_graph_gridworld(self, state_dictionary):

        rows, cols = 5,5
        num_grids = len(state_dictionary)

        colors = ['white', 'blue', 'red', 'green', 'green']
        cmap = ListedColormap(colors)
        # Define bounds for discrete colors
        norm = plt.Normalize(vmin=0, vmax=4)

        figs, axes = plt.subplots(1, len(state_dictionary), figsize=(num_grids * 3, 3), dpi=100)
        canvas = FigureCanvasTkAgg(figs, master=self.frame_graph)
        canvas.get_tk_widget().pack(fill='both', expand =True)



        ACTIONS = {'RIGHT': 0, 'DOWN': 1, 'LEFT': 2, 'UP': 3, 'CHOP': 4, 'SHOOT': 5}

        target_action = dict((v,k) for k,v in ACTIONS.items()).get(self.target)
        heading_text = f"What should the environment look like so it performs {target_action}"
        self.title_label.config(text=heading_text)



        # Add text labels for Agent (2) and Dragon (3)
        for i, (key,v)  in enumerate(state_dictionary.items()):
            # Reshape the 1D list into a 2D NumPy array (the grid)
            state_matrix = self.reshape_matrix(v[:25])


            ax = axes[i] if num_grids > 1 else axes  # Handle single subplot case''
            ax.imshow(state_matrix, cmap=cmap, norm=norm, origin='upper', interpolation='none')
            ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
            ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
            ax.tick_params(which='minor', size=0)
            ax.set_xticks([])
            ax.set_yticks([])
            for j in range(rows):
                for k in range(cols):
                    val = state_matrix[j, k]
                    if val == 1:
                        ax.text(k, j, 'A', ha='center', va='center', color='white', fontweight='bold', fontsize=16)
                    elif val == 2:
                        ax.text(k, j, 'M', ha='center', va='center', color='white', fontweight='bold', fontsize=16)

            ax.set_title(f'{key}')

        plt.tight_layout()
        canvas.draw()


    def generate_graph_frozen_lake(self, state_dictionary):
        rows, cols = 5, 5
        num_grids = len(state_dictionary)

        # Define colors for the grid values: 0=Safe Ice, 1=Hole, 2=Agent, 3=Goal
        # F: #ADD8E6 (Light Blue), H: #333333 (Dark Grey/Black), S: #008000 (Green), G: #FF4500 (Orange-Red)
        colors = ['#ADD8E6', '#333333', '#008000', '#FF4500']
        cmap = ListedColormap(colors)
        norm = plt.Normalize(vmin=0, vmax=3)  # Vmin=0, Vmax=3 for 4 categories

        # Create a figure and subplots
        figs, axes = plt.subplots(1, len(state_dictionary), figsize=(num_grids * 3, 3), dpi=100)
        canvas = FigureCanvasTkAgg(figs, master=self.frame_graph)
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Ensure axes is iterable even if there is only one subplot
        if num_grids == 1:
            axes = [axes]

            # Add text labels for Safe Ice (F), Hole (H), Agent (S), Goal (G)
        labels = {0: 'F', 1: 'H', 2: 'S', 3: 'G'}
        label_colors = {0: 'black', 1: 'white', 2: 'white', 3: 'white'}  # Text color

        ACTIONS = {'LEFT': 0, 'DOWN': 1, 'RIGHT': 2, 'UP': 3, 'EXIT': 4}
        target_action = dict((v, k) for k, v in ACTIONS.items()).get(self.target)
        heading_text = f"What should the environment look like so it performs {target_action}"
        self.title_label.config(text=heading_text)

        for i, (key, v) in enumerate(state_dictionary.items()):
            # Reshape the 7-element list into a 5x5 NumPy array (the grid)
            state_matrix = self.reshape_matrix_frozen_lake(v, rows, cols)

            ax = axes[i]
            ax.imshow(state_matrix, cmap=cmap, norm=norm, origin='upper', interpolation='none')

            # Grid lines
            ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
            ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
            ax.tick_params(which='minor', size=0)
            ax.set_xticks([])
            ax.set_yticks([])

            # Add text labels (F, H, S, G)
            for r in range(rows):
                for c in range(cols):
                    val = state_matrix[r, c]
                    label = labels.get(val, '')  # Get the label (F, H, S, G)
                    color = label_colors.get(val, 'black')  # Get the text color

                    # Special check: If the cell is the agent's current position (2),
                    # check if it's also the goal (3) in the original list.
                    is_goal = (self.index_to_coords(v[1]) == (r, c))  # Is the goal in this cell?

                    if val == 2 and not is_goal:
                        # Agent is at S, not G
                        display_label = 'S'
                    elif val == 3:
                        # Cell is the goal (G)
                        display_label = 'G'
                    elif val == 2 and is_goal:
                        # Agent is at the goal!
                        display_label = 'S/G'
                        # Using a mix of colors or a different color for S/G
                        color = 'yellow'
                    elif val == 1:
                        # Hole
                        display_label = 'H'
                    elif val == 0:
                        # Safe Ice
                        display_label = 'F'
                    else:
                        display_label = label  # Should catch H, G, F if S is elsewhere

                    ax.text(c, r, display_label, ha='center', va='center',
                            color=color, fontweight='bold', fontsize=16)

            ax.set_title(f'{key}')

        plt.tight_layout()
        canvas.draw()


    def index_to_coords(self, index, cols=5):
        """Converts a flat index (0-24) to (row, column) coordinates."""
        return index // cols, index % cols

    def reshape_matrix_frozen_lake(self, state_list, rows=5, cols=5):
        """
        Transforms the 7-element state list into a 5x5 matrix for plotting.
        The values in the matrix represent:
        0: Safe Ice (F)
        1: Hole (H)
        2: Agent (S)
        3: Goal (G)
        """
        # Initialize an empty 5x5 grid with safe ice (0)
        grid = np.zeros((rows, cols), dtype=int)

        # State list: [Agent (0), Goal (1), Hole1 (2), H2 (3), H3 (4), H4 (5), H5 (6)]

        agent_pos = state_list[0]
        goal_pos = state_list[1]
        hole_positions = state_list[2:]

        # 1. Place Holes (Value 1)
        for hole_index in hole_positions:
            r, c = self.index_to_coords(hole_index, cols)
            if 0 <= r < rows and 0 <= c < cols:
                grid[r, c] = 1

        # 2. Place Goal (Value 3)
        gr, gc = self.index_to_coords(goal_pos, cols)
        if 0 <= gr < rows and 0 <= gc < cols:
            grid[gr, gc] = 3

        # 3. Place Agent (Value 2) - This overrides the initial 0 or 1 if agent is in a hole
        ar, ac = self.index_to_coords(agent_pos, cols)
        if 0 <= ar < rows and 0 <= ac < cols:
            # NOTE: If the agent is in the goal, the 'G' label will be plotted over the 'S' label
            # If the agent is in a hole, the 'S' label will be plotted over the 'H' label
            grid[ar, ac] = 2

        return grid






    def reshape_matrix(self, state_list):
        try:
            state_matrix = np.array(state_list, dtype=int).reshape(5, 5)
            return state_matrix
        except ValueError:
            print(
                f"Error: List length ({len(state_list)}) does not match grid dimensions ({5}x{5}={25}).")
            return None




if __name__ == "__main__":
    root = tk.Tk()
    app = CSVGraphApp(root)
    root.mainloop()