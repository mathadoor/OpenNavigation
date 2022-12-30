# Import Relevant Libraries
import habitat
import clip
import torch
import numpy as np
from gym import spaces
from habitat_baselines.config.default import get_config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.config.default_structured_configs import ObjectGoalSensorConfig
from habitat.core.registry import registry

DATA_DIR = "/home/paperspace/Documents/habitat_project/habitat-lab_call_for_collab/data/datasets/imagenav/train/content/"
RUN_TYPE = 'train'
CONFIG_FILE = "./habitat-lab_call_for_collab/habitat-baselines/habitat_baselines/config/imagenav/ddppo_imagenav_gibson.yaml"
OPT = [
    "habitat.dataset.data_path='/home/paperspace/Documents/habitat_project/habitat-lab_call_for_collab/data/datasets/imagenav/train/train.json.gz'",
    "habitat_baselines.num_environments=2",
    "habitat.dataset.content_scenes=['1S7LAXRdDqK']",
    "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=256",
    "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=256",
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=256",
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=256",
    "habitat_baselines.load_resume_state_config=False"]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@registry.register_sensor(name="CaptionGoalSensor")
class CaptionGoalSensor(Sensor):
    def __init__(self, sim, config, dataset, **kwargs):
        self._sim = sim
        self._dataset = dataset
        self.device = DEVICE
        self.model, preprocess = clip.load("RN50")
        self.model.to(self.device)
        super().__init__(config=config)

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(1024,), dtype=np.float32
        )

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args, **kwargs):
        return "caption_goal_embedding"

    def get_observation(self, observations, *args, episode, **kwargs):
        # caption = episode['info']['CaptionGoal']
        try:
            caption = episode.info['CaptionGoal']
        except NameError:
            print(f"Caption Goal Does Not Exist in Episode: {episode.episode_id}")
        caption_token = clip.tokenize(caption).to(self.device)
        return self.model.encode_text(caption_token).detach().float().squeeze()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    config = get_config(CONFIG_FILE, OPT)

    with habitat.config.read_write(config):
        config.habitat_baselines["num_updates"] = -1
        config.habitat_baselines["total_num_steps"] = 1000000
        config.habitat.task.lab_sensors["CaptionGoalSensor"] = ObjectGoalSensorConfig(type="CaptionGoalSensor")
    trainer_init = baseline_registry.get_trainer(config.habitat_baselines.trainer_name)
    trainer = trainer_init(config)
    trainer.train()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
