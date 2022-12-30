# Import Relevant Libraries
import torch
import json
import gzip
from tqdm import tqdm
from habitat_baselines.config.default import get_config
from habitat.core.env import Env
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

DATA_DIR = "/home/paperspace/Documents/habitat_project/habitat-lab_call_for_collab/data/datasets/imagenav/train/content/"
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
MAX_LENGTH = 16
NUM_BEAMS = 4


def update_dataset(env_: Env, type="vit-gpt2", device="cpu"):
    gen_kwargs = {"max_length": MAX_LENGTH, "num_beams": NUM_BEAMS}
    if type == "vit-gpt2":
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        model.to(device)
        task = env_.task
        with gzip.open(DATA_DIR + env_.sim.curr_scene_name + '.json.gz') as f:
            json_scene = json.loads(f.read())
            for i in tqdm(range(len(env_.episodes))):
                episode = env_.episodes[i]
                obs = task.reset(episode)
                episode_id = episode.episode_id
                image = obs["imagegoal"]
                pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(device)

                output_ids = model.generate(pixel_values, **gen_kwargs)
                json_scene['episodes'][int(episode_id)]['info']['CaptionGoal'] =\
                tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            f.close()

        with gzip.open(DATA_DIR + env_.sim.curr_scene_name + '.json.gz', "wt") as f:
            string_data = json.dumps(json_scene)
            f.write(string_data)
            f.close()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    config = get_config(CONFIG_FILE, OPT)
    environment = Env(config)
    update_dataset(environment, device=DEVICE)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
