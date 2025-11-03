import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import numpy as np
import re
from pathlib import Path


def visualize_matplotlib_slider_plan(solution_file: Path):
    """
    Visualizes a plan from a solution file using Matplotlib with a slider.

    Args:
        solution_file (str): The path to the solution file.
    """

    # --- 1. Parse and read all data files (similar to previous versions) ---
    try:
        basename = solution_file.stem
        parts = basename.split("-")
        map_name = parts[1]
        test_num = parts[-1].replace("test", "")
        obs_num_str = parts[-2].replace("obs", "")

        map_file = Path("maps") / f"{map_name}.txt"
        test_file = (
            Path("tests") / map_name / f"{map_name}-test-{test_num}-{obs_num_str}.txt"
        )

    except IndexError:
        print(f"Error: Could not parse filename: {solution_file}")
        return

    try:
        print(f"{map_file=}")
        with map_file.open() as f:
            H, W = map(int, f.readline().split())
            grid_data = np.array([list(map(int, line.split())) for line in f])
    except Exception as e:
        print(f"Error reading map file: {e}")
        return

    try:
        F = 10
        print(f"{test_file=}")
        with test_file.open() as f:
            lines = f.readlines()
        st_x = int(re.search(r"x=\s*(\d+)", lines[0]).group(1))
        st_y = int(re.search(r"y=\s*(\d+)", lines[0]).group(1))
        end_x = int(re.search(r"x=\s*(\d+)", lines[1]).group(1))
        end_y = int(re.search(r"y=\s*(\d+)", lines[1]).group(1))
        dynamic_obstacles = []
        for line in lines[2:]:
            match = re.search(
                r"x=\s*(\d+)\s*y=\s*(\d+)\s*from_timestep:\s*(\d+)\s*to_timestep:\s*(\d+)",
                line,
            )
            if match:
                dyn_x, dyn_y, t1, t2 = map(int, match.groups())
                dynamic_obstacles.append((dyn_x, dyn_y, t1 * F, t2 * F))
    except Exception as e:
        print(f"Error reading or parsing test file: {e}")
        return

    try:
        plan_df = pd.read_csv(solution_file)
    except Exception as e:
        print(f"Error reading solution file: {e}")
        return

    # --- 2. Prepare data for animation ---
    # The new solution file format is already t,x,y per timestep.
    agent_path_df = plan_df.set_index("t")
    max_time = agent_path_df.index.max()


    # --- 3. Setup the plot ---
    fig, ax = plt.subplots(figsize=(10, 10 * H / W))
    plt.subplots_adjust(bottom=0.25)

    ax.imshow(
        grid_data,
        cmap=plt.cm.colors.ListedColormap(["white", "black"]),
        origin="lower",
        interpolation="none",
    )
    ax.plot(st_y, st_x, "go", markersize=10, label="Start")
    ax.plot(end_y, end_x, "ro", markersize=10, label="End")
    # Use agent_path_df for the full path plot
    ax.plot(agent_path_df["y"], agent_path_df["x"], "b--", alpha=0.3, label="Full Path")

    (agent_plot,) = ax.plot([], [], "bo", markersize=6, label="Agent")
    dyn_obs_plots = ax.scatter(
        [], [], s=5, c="orange", marker="s", label="Dynamic Obstacle"
    )
    time_text = ax.text(
        0.02,
        0.02,
        "",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    ax.set_title(f"Plan for {map_name}")
    ax.set_xlabel("Y")
    ax.set_ylabel("X")
    ax.set_xticks(np.arange(0, W, 1))
    ax.set_yticks(np.arange(0, H, 1))
    ax.grid(which="both", color="gray", linestyle="-", linewidth=0.5)
    ax.legend()
    ax.invert_yaxis()

    # --- 4. Define update function for animation and slider ---
    def update_frame(t):
        t = int(t)
        # Use .get(t, default=None) or check if t is in index to avoid KeyError
        if t in agent_path_df.index:
            agent_pos = agent_path_df.loc[t]
            agent_plot.set_data([agent_pos["y"]], [agent_pos["x"]])

        dyn_obs_plots.set_offsets(np.empty((0, 2)))

        active_obs_coords = []
        for x, y, t1, t2 in dynamic_obstacles:
            if t1 <= t <= t2:
                active_obs_coords.append((y, x))

        if active_obs_coords:
            dyn_obs_plots.set_offsets(np.array(active_obs_coords))

        time_text.set_text(f"Time: {t}")
        fig.canvas.draw_idle()

    # --- 5. Create Widgets (Slider and Buttons) ---
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    time_slider = Slider(
        ax=ax_slider,
        label="Time",
        valmin=0,
        valmax=max_time, # Use calculated max_time
        valinit=0,
        valstep=1,
    )
    time_slider.on_changed(update_frame)

    # Animation object
    ani = animation.FuncAnimation(
        fig,
        update_frame,
        frames=range(max_time + 1), # Use calculated max_time
        blit=False,
        interval=50,
        repeat=False,
    )
    ani.pause()  # Start paused

    # Play/Pause buttons
    class Player:
        def __init__(self, ani, slider):
            self.ani = ani
            self.slider = slider
            self.is_playing = False

        def play(self, event):
            if not self.is_playing:
                self.is_playing = True
                self.ani.resume()

        def pause(self, event):
            if self.is_playing:
                self.is_playing = False
                self.ani.pause()

        def update_slider(self, event):
            self.slider.set_val(event)

    ax_play = plt.axes([0.8, 0.025, 0.07, 0.04])
    ax_pause = plt.axes([0.7, 0.025, 0.07, 0.04])
    btn_play = Button(ax_play, "Play")
    btn_pause = Button(ax_pause, "Pause")

    player = Player(ani, time_slider)
    btn_play.on_clicked(player.play)
    btn_pause.on_clicked(player.pause)
    # Link animation frames to slider
    ani._func = lambda frame: player.update_slider(frame)

    update_frame(0)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animate a SIPP-IP plan using Matplotlib with a slider."
    )
    parser.add_argument(
        "solution_file",
        type=Path,
        help="Path to the solution file (e.g., results/solutions/sol-....txt)",
    )
    args = parser.parse_args()

    visualize_matplotlib_slider_plan(args.solution_file)
