# Import Relevant Libraries
import habitat
import clip
import torch
import numpy as np
from gym import spaces
from habitat_baselines.config.default import get_config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
from habitat.core.simulator import Sensor, SensorTypes
from habitat.config.default_structured_configs import ObjectGoalSensorConfig
from habitat.core.registry import registry

RUN_TYPE = 'train'
CONFIG_FILE = "./habitat-lab_call_for_collab/habitat-baselines/habitat_baselines/config/objectnav" \
              "/ddppo_objectnav_hm3d.yaml"
OPT = [
    "habitat.dataset.data_path='/content/habitat-lab_call_for_collab/data/datasets/objectnav/objectnav_hm3d_v1/train/train.json.gz'",
      "habitat_baselines.num_environments=16",
      "habitat.dataset.scenes_dir='/content/habitat-lab_call_for_collab/data/scene_datasets'",
      "habitat.dataset.content_scenes=['1S7LAXRdDqK']",
      "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=256",
      "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=256",
      "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=256",
      "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=256",
      "habitat_baselines.load_resume_state_config=False",]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
text_embeddings = {}

def get_obj_prompt(config, load_data=False):
    # load_data is used in case the user is unaware of the exact name of the categories in the database.
    if load_data:
        data = habitat.datasets.object_nav.object_nav_dataset.ObjectNavDatasetV1(config.habitat.dataset)
        object_categories = list(data.category_to_task_category_id.keys())
        del data
    else:
        object_categories = ["chair", "sofa", "plant", "bed", "toilet", "tv_monitor"]
    # Convert object categories into object prompts
    object_prompts = ["a photo of " + category for category in object_categories]
    return object_categories, object_prompts


def get_embeddings(object_categories, object_prompts, device='cpu'):
    # Load CLIP model, convert the prompts, and store them as a dictionary
    model, preprocess = clip.load("RN50")
    model.to(device)
    text_tokens = clip.tokenize(object_prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()

    for k, v in zip(object_categories, text_features):
        text_embeddings[k] = v

    return text_embeddings

# Register ObjectGoalPromptSensor
@registry.register_sensor(name="ObjectGoalPromptSensor")
class ObjectGoalPromptSensor(Sensor):
    def __init__(self, sim, config, dataset, **kwargs):
        self._sim = sim
        self._dataset = dataset
        self._embeddings = torch.load('clip_embeddings.pt')
        super().__init__(config=config)

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(1024,), dtype=np.float32
        )

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args, **kwargs):
        return "goal_prompt_embedding"

    def get_observation(self, observations, *args, episode, **kwargs):
        category_name = episode.object_category
        return self._embeddings[category_name]


if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Load Config
    config = get_config(CONFIG_FILE, OPT)

    # Get Object Data:
    cats, prompts = get_obj_prompt(config, load_data=False)
    text_embeddings = get_embeddings(cats, prompts, device=DEVICE)
    torch.save(text_embeddings, 'clip_embeddings.pt')
    
    # Add ObjectGoalPromptSensor to the Config
    with habitat.config.read_write(config):
      
    # Now define the config for the sensor
        config.habitat_baselines["num_updates"] = -1
        config.habitat_baselines["total_num_steps"] = 1000000
        config.habitat.task.lab_sensors["ObjectGoalPromptSensor"] = ObjectGoalSensorConfig(type="ObjectGoalPromptSensor")
        del config.habitat.task.lab_sensors["objectgoal_sensor"]

    # Initialize and Run Training
    trainer_init = baseline_registry.get_trainer(config.habitat_baselines.trainer_name)
    trainer = trainer_init(config)
    trainer.train()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
