# OpenNavigation

The repository contains the files used in the response towards the spring 2023 call for collaboration by CV MLP lab at Georgia Tech. 
The descriptions of the files are as follows:

- prompt_run.py: python script containing the code to export ObjectGoal embeddings, define GoalPromptSensor, and running the baseline with the code. In addition, habitat_baselines/rl/ddppo/policy/resnet_policy.py is also updated to include the embeddings in the [initialization](https://github.com/mathadoor/habitat-lab_call_for_collab/blob/4fb1ab0b408a9f2f9b62f1bca81e178a1ce37672/habitat-baselines/habitat_baselines/rl/ddppo/policy/resnet_policy.py#L370) and the [forward](https://github.com/mathadoor/habitat-lab_call_for_collab/blob/4fb1ab0b408a9f2f9b62f1bca81e178a1ce37672/habitat-baselines/habitat_baselines/rl/ddppo/policy/resnet_policy.py#L549) functions of the policy network.

- dataset_captions_addition.py: python script used to load the image goal of each episode, generate the corresponding caption, and update the dataset. Note the script currently uses [ViT-GPT2](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) encoder-decoder architecture from HuggingFace to generate the captions.

- caption_run.py: python script to run the baseline with CaptionGoalSensor that encodes the captions with CLIP text encoder. The overall structure is similar to that of prompt_run.py.

In addition, you can access the Google Colab Notebooks I setup for the GoalPrompt task, and the CaptionNav task at the following links: [GoalPromptTask](https://colab.research.google.com/drive/1TmHk7KI2G0G4zp60IhLC6erMW9829hEv?usp=sharing) and [CaptionNavTask](https://colab.research.google.com/drive/1YbTJv4TK9KTwVw9hlgYm6MPBRqdrRgai?usp=sharing)

If you have any questions, please feel free to reach out to me at matharooh2@gmail.com or hmatharoo3@gatech.edu.

