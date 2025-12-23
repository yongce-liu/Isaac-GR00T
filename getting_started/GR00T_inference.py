# %% [markdown]
# # GR00T Inference
#
# This tutorial shows how to use the GR00T inference model to predict the actions from the observations, given a test dataset.

# %%
import os

import gr00t
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
import torch


# %%
# change the following paths
MODEL_PATH = "nvidia/GR00T-N1.6-3B"

# REPO_PATH is the path of the pip install gr00t repo and one level up
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
DATASET_PATH = os.path.join(REPO_PATH, "demo_data/gr1.PickNPlace")
EMBODIMENT_TAG = "gr1"

device = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## Loading Pretrained Policy
#
# Policy Model is loaded just like any other huggingface model.
#
# There are 2 new concepts here in the GR00T model:
#  - modality config: This defines the keys in the dictionary used by the model. (e.g. `action`, `state`, `annotation`, `video`)
#  - modality_transform: A sequence of transform which are used during dataloading

# %%
policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    embodiment_tag=EmbodimentTag(EMBODIMENT_TAG),
    device=device,
    strict=True,
)

# print out the policy model architecture
print(policy.model)

# %% [markdown]
# ## Loading dataset

# %% [markdown]
# First this requires user to check which embodiment tags are used to pretrained the `Gr00tPolicy` pretrained models.

# %%
import numpy as np


modality_config = policy.get_modality_config()

print(modality_config.keys())

for key, value in modality_config.items():
    if isinstance(value, np.ndarray):
        print(key, value.shape)
    else:
        print(key, value)


# %%
# Create the dataset
dataset = LeRobotEpisodeLoader(
    dataset_path=DATASET_PATH,
    modality_configs=modality_config,
    video_backend="torchcodec",
    video_backend_kwargs=None,
)

# %% [markdown]
# Let's print out a single data and visualize it

# %%
import numpy as np


episode_data = dataset[0]
step_data = extract_step_data(
    episode_data,
    step_index=0,
    modality_configs=modality_config,
    embodiment_tag=EmbodimentTag(EMBODIMENT_TAG),
    allow_padding=False,
)

# print(step_data)

print("\n\n====================================")
print("Images:")
for img_key in step_data.images:
    print(
        " " * 4, img_key, f"{len(step_data.images[img_key])} x {step_data.images[img_key][0].shape}"
    )

print("\nStates:")
for state_key in step_data.states:
    print(" " * 4, state_key, step_data.states[state_key].shape)

print("\nActions:")
for action_key in step_data.actions:
    print(" " * 4, action_key, step_data.actions[action_key].shape)

print("\nTask: ", step_data.text)


# %% [markdown]
# Let's plot just the "right arm" state and action data and see how it looks like. Also show the images of the right hand state.

# %%
import matplotlib.pyplot as plt


episode_index = 0
max_steps = 400
joint_name = "right_arm"
image_key = "ego_view_bg_crop_pad_res256_freq20"

state_joints_across_time = []
gt_action_joints_across_time = []
images = []

sample_images = 6
episode_data = dataset[episode_index]
print(len(episode_data))

for step_count in range(max_steps):
    data_point = extract_step_data(
        episode_data,
        step_index=step_count,
        modality_configs=modality_config,
        embodiment_tag=EmbodimentTag(EMBODIMENT_TAG),
        allow_padding=False,
    )
    state_joints = data_point.states[joint_name][0]
    gt_action_joints = data_point.actions[joint_name][0]

    state_joints_across_time.append(state_joints)
    gt_action_joints_across_time.append(gt_action_joints)

    # We can also get the image data
    if step_count % (max_steps // sample_images) == 0:
        image = data_point.images[image_key][0]
        images.append(image)

# Size is (max_steps, num_joints)
state_joints_across_time = np.array(state_joints_across_time)
gt_action_joints_across_time = np.array(gt_action_joints_across_time)


# Plot the joint angles across time
num_joints = state_joints_across_time.shape[1]
fig, axes = plt.subplots(nrows=num_joints, ncols=1, figsize=(8, 2 * num_joints))

for i, ax in enumerate(axes):
    ax.plot(state_joints_across_time[:, i], label="state joints")
    ax.plot(gt_action_joints_across_time[:, i], label="gt action joints")
    ax.set_title(f"Joint {i}")
    ax.legend()

plt.tight_layout()
plt.show()


# Plot the images in a row
fig, axes = plt.subplots(nrows=1, ncols=sample_images, figsize=(16, 4))

for i, ax in enumerate(axes):
    ax.imshow(images[i])
    ax.axis("off")

# %% [markdown]
# Now we can run the policy from the pretrained checkpoint.

# %%
observation = {
    "video": {
        k: np.stack(step_data.images[k])[None] for k in step_data.images
    },  # stach images and add batch dimension
    "state": {k: step_data.states[k][None] for k in step_data.states},  # add batch dimension
    "action": {k: step_data.actions[k][None] for k in step_data.actions},  # add batch dimension
    "language": {
        modality_config["language"].modality_keys[0]: [
            [step_data.text]
        ],  # add time and batch dimension
    },
}
predicted_action, info = policy.get_action(observation)
for key, value in predicted_action.items():
    print(key, value.shape)

# %% [markdown]
# For more details on the policy (e.g. expected input and output), please refer to the [policy documentation](policy.md).
