# How to use GR00T-N1.5

## Dataset
- load dataset
```python
dataset = gr00t.data.dataset.LeRobotSingleDataset(
            dataset_path: Path | str,
            modality_configs: dict[str, ModalityConfig],
            embodiment_tag: str | EmbodimentTag,
            video_backend: str = "torchcodec",
            video_backend_kwargs: dict | None = None,
            transforms: ComposedModalityTransform | None = None,
        )
```
- get data
```python
resp: dict[str, any] = dataset[i]
```

## Inference
- get data -> resp
- get policy
```python
policy = gr00t.model.policy.Gr00tPolicy(
            model_path=MODEL_PATH,
            embodiment_tag=EMBODIMENT_TAG,
            modality_config=modality_config, # the modality you used in downstream, e.g., [state, action, video, ...]
            modality_transform=modality_transform, # the transdorm you used for the dataset
            device=device,
        )
```
- inference: get the action given the observations
```python
pred_action: dict[str, any] = policy.get_action(resp)
```
### An example for the pred_action
```python
action.left_arm (16, 7) # 16 predict horizon, 7 dof arm
action.right_arm (16, 7) 
action.left_hand (16, 6) # 16 predict horizon, 6 dof hand
action.right_hand (16, 6)
```

## Finetune
- get the new data, e.g., ```../datasets/so101-table-cleanup```
- use ```LeRobotSingleDataset``` to load the data
- use ```Gr00tPolicy``` to load the pretrained policy
- use ```scripts/gr00t_finetune.py``` to finetune, e.g.,
```bash
python scripts/gr00t_finetune.py \
   --dataset-path datasets/so101-table-cleanup/ \
   --num-gpus 1 \
   --batch-size 64 \
   --output-dir so101-checkpoints  \
   --max-steps 10000 \
   --data-config so100_dualcam
```