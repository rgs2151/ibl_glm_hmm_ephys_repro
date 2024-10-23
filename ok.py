# %%
from manim import *
from utils import download_glm_hmm, save_data_path, val_eid, all_eid
from one.api import ONE
one = ONE(password='international')

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# %%
eid = val_eid[0]

# %% [markdown]
# ```Original # Frame rate: 60.0, Dimensions: 1280x1024```

# %%
def get_frames(video_path, start_frame, end_frame):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = end_frame - start_frame
    frames = np.empty((total_frames, 1024, 1280, 3), dtype=np.uint8)  # Pre-allocate with fixed size

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames[i] = frame
    cap.release()

    return frames

# %%
def load_viz_data(eid, trial_idx):

    tr = one.load_object(eid, 'trials')

    tr_interval = tr['intervals'][trial_idx]
    stim_on = tr['stimOn_times'][trial_idx]
    stim_off = tr['stimOff_times'][trial_idx]
    go_cue = tr['goCue_times'][trial_idx]
    feeback_time = tr['feedback_times'][trial_idx]

    feedback_type = tr["feedbackType"][trial_idx]
    
    camera_times = one.load_dataset(eid, f'*leftCamera.times.npy', collection='alf')

    # closest camera times to trial start and end
    camera_start_idx = np.searchsorted(camera_times, tr_interval[0], side='left')
    camera_end_idx = np.searchsorted(camera_times, tr_interval[1], side='right')


    video_path = one.eid2path(eid).joinpath("raw_video_data/_iblrig_leftCamera.raw.mp4")
    frames = get_frames(video_path, camera_start_idx, camera_end_idx)


    frame_pred_path = Path("data") / "frame_probs" / f"{eid}_probs.csv"
    probabilities = pd.read_csv(frame_pred_path).to_numpy()[camera_start_idx:camera_end_idx]

    timeline_data = (tr_interval, stim_on, stim_off, go_cue, feeback_time)
    
    return camera_start_idx, camera_end_idx, camera_times, frames, probabilities, timeline_data, feedback_type


# %%
def make_legend(position = UP * 2.5 + RIGHT * 1):
    
    # Create the legend
    labels_legend = ["still", "move", "wheel_turn", "groom"]
    colors_legend = [RED, BLUE, GREEN, YELLOW]
    legend_items = VGroup()

    for label_text, color in zip(labels_legend, colors_legend):
        legend_marker = Square(
            side_length=0.1,
            fill_color=color,
            fill_opacity=1,
            stroke_color=color
        )
        legend_label = Text(label_text, font_size=12)
        legend_entry = VGroup(legend_marker, legend_label)
        legend_entry.arrange(RIGHT, buff=0.2)
        legend_items.add(legend_entry)

    legend_items.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
    legend_items.height = 1

    legend_items.move_to(position)

    return legend_items


# %%
def make_image(idx, 
               frames, 
               position = UP * 2 + RIGHT * 4):
    
    image = ImageMobject(frames[idx], image_mode="BGR")
    image.height = 3
    image.move_to(position)
    return image

# %%

def make_graph(idx, 
               x, 
               y, 
               colors = None,
               position = UP * 2.5 + LEFT * 3.5):
    """
    Create a graph with the given n lines using x and y values
    
    Parameters:
    idx : int
        Index of the frame
    x : (lines, x_values)
        list of lists of x values
    y : (lines, y_values)
        list of lists of y values
    static_axis : bool
        If True, the x-axis will be static, otherwise it will be dynamic

    Returns:
    graph_group : VGroup
        Group of graph, axes and labels
    """
    
    ax = Axes(x_range=[x[0][0], x[0][idx], max(x[0][idx]/10,1)],
              y_range=[0, 1, 0.2],
              x_length=12,
              y_length=3,
              axis_config={"include_numbers": True},
              tips=False,
              )

    labels = ax.get_axis_labels(x_label='t (s)', y_label='p(frame)')

    lines_arr = []
    for l in range(len(x)):
        line = ax.plot_line_graph(x_values=np.array(x[l]), 
                                y_values=np.array(y[l]), 
                                add_vertex_dots=False,
                                stroke_width=2,
                                line_color= colors[l] if colors else BLUE,
                                )
        lines_arr.append(line)

    # Group axes and labels
    graph_group = VGroup(ax, labels, *lines_arr)
    graph_group.height = 2
    graph_group.move_to(position)
    
    return graph_group

# %%
def make_timeline(timeline_data,
                  play_head_position = None,
                  marker_spacing=1, 
                  position = UP * 1 + RIGHT * 4):
    
    tr_interval, stim_on, stim_off, go_cue, feedback_time = timeline_data
    
    # Create the number line
    number_line = NumberLine(x_range=[tr_interval[0], tr_interval[1], 2],
                             length=10,
                             color=WHITE,
                             include_numbers=True,
                             label_direction=UP)
    
    # Add vertical lines for events and labels
    stim_on_line = Arrow(
        start=number_line.n2p(stim_on) + UP * marker_spacing,
        end=number_line.n2p(stim_on)
    )
    stim_off_line = Arrow(
        start=number_line.n2p(stim_off) + UP * marker_spacing,
        end=number_line.n2p(stim_off)
    )
    go_cue_line = Arrow(
        start=number_line.n2p(go_cue) + DOWN * marker_spacing,
        end=number_line.n2p(go_cue)
    )
    feedback_line = Arrow(
        start=number_line.n2p(feedback_time) + DOWN * marker_spacing,
        end=number_line.n2p(feedback_time)
    )
    
    # Create labels
    stim_on_label = Text("stimOn", font_size=18).next_to(stim_on_line, UP)
    stim_off_label = Text("stimOff", font_size=18).next_to(stim_off_line, UP)
    go_cue_label = Text("goCue", font_size=18).next_to(go_cue_line, DOWN)
    feedback_label = Text("feedback", font_size=18).next_to(feedback_line, DOWN)
    
    # Playhead indicator
    playhead = Dot(number_line.n2p(tr_interval[0]) if play_head_position is None else number_line.n2p(play_head_position),
                   radius=.1,
                   color=RED)

    number_group = VGroup(number_line, stim_on_line, stim_off_line, go_cue_line, feedback_line)
    label_group = VGroup(stim_on_label, stim_off_label, go_cue_label, feedback_label)

    timeline_group = VGroup(number_group, label_group, playhead)

    timeline_group.move_to(position)

    return timeline_group

class InspectMouse(Scene):

   def construct(self):

      camera_start_idx, camera_end_idx, camera_times, frames, probabilities, timeline_data, feedback_type = load_viz_data(eid, trial_idx = 1)
      
      # Initiate empty elements
      image = ImageMobject(np.zeros((1024, 1280, 3)))
      graph = VGroup()
      legend = make_legend()
      timeline = VGroup()
      
      self.add(image, graph, legend, timeline)

      # ValueTracker for timestep
      timestep = ValueTracker(0)

      # Updater for image
      def update_image(mob):
         idx = int(timestep.get_value())
         mob.become(make_image(idx, frames))
      
      # Updater for graph
      def update_prob_graph(mob):
         idx = int(timestep.get_value())
         x = np.array([camera_times[:idx + 1]]*4)
         y = probabilities[:idx].T
         mob.become(make_graph(idx, x, y, colors=[RED, BLUE, GREEN, YELLOW]))

      def update_timeline(mob):
         idx = int(timestep.get_value())
         mob.become(make_timeline(timeline_data, play_head_position=camera_times[idx]))
      
      image.add_updater(update_image)
      graph.add_updater(update_prob_graph)

      # Animation duration based on number of frames
      self.play(timestep.animate.set_value(60), run_time=1, rate_func=linear)
      
      # Remove updaters after animation
      image.remove_updater(update_image)
      graph.remove_updater(update_prob_graph)


