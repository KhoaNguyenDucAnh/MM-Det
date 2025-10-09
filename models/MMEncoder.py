import os

import lightning as L
import numpy as np
import torch
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from PIL import Image
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig


class MMEncoder(L.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.interval = config["interval"]
        self.new_tokens = config["new_tokens"]
        self.selected_layers = config["selected_layers"]

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=config["lmm_ckpt"],
            model_base=config["lmm_base"],
            load_8bit=config["load_8bit"],
            load_4bit=config["load_4bit"],
            model_name=get_model_name_from_path(config["lmm_ckpt"]),
        )
        self.tokenizer.chat_template = """{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"""

    def get_prompt(self, image):
        return [
            {
                "role": "system",
                "content": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": "As an expert in image forensics, you are to briefly describe the image, including lighting and reflection, texture, color saturation, shape consistency, sense of depth, compression trace, artifacts. Give a reason to justify whether it is a real or a fake image.",
                    },
                ],
            },
        ]

    def predict_step(self, batch):
        mm_representation = {}
        for video_path, extracted_frames in batch:
            sample_index = list(range(0, extracted_frames.shape[0], self.interval))
            visual_features_list = []
            textual_features_list = []
            for frame in extracted_frames[sample_index]:
                visual_features, textual_features = self.forward(frame)
                visual_features_list.append(visual_features.detach().cpu().numpy())
                textual_features_list.append(textual_features.detach().cpu().numpy())
            mm_representation[os.path.join("visual", video_path)] = np.array(
                visual_features_list
            )
            mm_representation[os.path.join("textual", video_path)] = np.array(
                textual_features_list
            )
        return mm_representation

    def forward(self, image):
        input_ids = self.tokenizer.apply_chat_template(
            self.get_prompt(image),
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self.device)

        processed_image = self.image_processor(images=image, return_tensors="pt").to(
            self.device
        )["pixel_values"]
        processed_image = processed_image.half()
        visual_features = self.model.model.vision_tower.vision_tower(
            processed_image
        ).pooler_output

        output = self.model.generate(
            input_ids,
            images=processed_image,
            do_sample=False,
            min_new_tokens=self.new_tokens,
            max_new_tokens=self.new_tokens,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # print(self.tokenizer.decode(output.sequences[0]))

        textual_features = []
        for layer_index in self.selected_layers:
            hidden_feature = [
                hidden_state[layer_index] for hidden_state in output["hidden_states"]
            ]
            hidden_feature = torch.cat(hidden_feature, dim=1)
            hidden_feature = hidden_feature[:, -self.new_tokens :, :]
            textual_features.append(hidden_feature)
        textual_features = torch.cat(textual_features, dim=0)
        return visual_features, textual_features


# class MMEncoderBase(L.LightningModule):
#     def __init__(self, config):
#         super().__init__()
#         self.interval = config["interval"]

#     def predict_step(self, batch):
#         mm_representation = {}
#         for video_path, extracted_frames in batch:
#             sample_index = list(range(0, extracted_frames.shape[0], self.interval))
#             visual_features_list = []
#             textual_features_list = []
#             for frame in extracted_frames[sample_index]:
#                 visual_features, textual_features = self.forward(frame)
#                 visual_features_list.append(visual_features.detach().cpu().numpy())
#                 textual_features_list.append(textual_features.detach().cpu().numpy())
#             mm_representation[os.path.join("visual", video_path)] = np.array(
#                 visual_features_list
#             )
#             mm_representation[os.path.join("textual", video_path)] = np.array(
#                 textual_features_list
#             )
#         return mm_representation


# class MMEncoder(MMEncoderBase):
#     def __init__(self, config, **kwargs):
#         super().__init__(config)

#         self.new_tokens = config["new_tokens"]
#         self.selected_layers = config["selected_layers"]

#         self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
#             model_path=config["lmm_ckpt"],
#             model_base=config["lmm_base"],
#             load_8bit=config["load_8bit"],
#             load_4bit=config["load_4bit"],
#             model_name=get_model_name_from_path(config["lmm_ckpt"]),
#         )
#         self.tokenizer.chat_template = """{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"""

#     def get_prompt(self, image):
#         return [
#             {
#                 "role": "system",
#                 "content": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": image},
#                     {
#                         "type": "text",
#                         "text": "As an expert in image forensics, you are to briefly describe the image, including lighting and reflection, texture, color saturation, shape consistency, sense of depth, compression trace, artifacts. Give a reason to justify whether it is a real or a fake image.",
#                     },
#                 ],
#             },
#         ]

#     def forward(self, X):
#         image = X
#         self.model = self.model.to(self.device)

#         input_ids = self.tokenizer.apply_chat_template(
#             self.get_prompt(image),
#             add_generation_prompt=True,
#             tokenize=True,
#             return_tensors="pt",
#         ).to(self.device)

#         processed_image = self.image_processor(images=image, return_tensors="pt").to(
#             self.device
#         )["pixel_values"]
#         processed_image = processed_image.half()
#         visual_features = self.model.model.vision_tower.vision_tower(
#             processed_image
#         ).pooler_output

#         output = self.model.generate(
#             input_ids,
#             images=processed_image,
#             do_sample=False,
#             min_new_tokens=self.new_tokens,
#             max_new_tokens=self.new_tokens,
#             use_cache=True,
#             output_hidden_states=True,
#             return_dict_in_generate=True,
#         )

#         # print(self.tokenizer.decode(output.sequences[0]))

#         textual_features = []
#         for layer_index in self.selected_layers:
#             hidden_feature = [
#                 hidden_state[layer_index] for hidden_state in output["hidden_states"]
#             ]
#             hidden_feature = torch.cat(hidden_feature, dim=1)
#             hidden_feature = hidden_feature[:, -self.new_tokens :, :]
#             textual_features.append(hidden_feature)
#         textual_features = torch.cat(textual_features, dim=0)
#         return visual_features, textual_features


# # pipenv update transformers
# class MMEncoder2(MMEncoderBase):
#     def __init__(self, config, **kwargs):
#         super().__init__(config)

#         self.new_tokens = config["new_tokens"]
#         self.selected_layers = config["selected_layers"]

#         self.processor = AutoProcessor.from_pretrained(config["lmm_ckpt"])
#         self.image_processor = self.processor.image_processor

#         quantization_config = BitsAndBytesConfig(
#             load_in_8bit=config["load_8bit"],
#             load_in_4bit=config["load_4bit"],
#         )

#         self.model = AutoModel.from_pretrained(
#             config["lmm_ckpt"], quantization_config=quantization_config
#         )

#     def get_prompt(self, image):
#         return [
#             {
#                 "role": "system",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": (
#                             "You are a highly skilled image forensics expert."
#                             "Your task is to critically analyze visual content for signs of digital manipulation, inconsistencies, or deepfake generation."
#                         ),
#                     }
#                 ],
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": image},
#                     {
#                         "type": "text",
#                         "text": (
#                             "Please conduct a detailed forensic analysis of the provided image. Evaluate the following aspects:\n\n"
#                             "- **Logical consistency**: Does the overall scene make sense contextually and physically?\n"
#                             "- **Lighting and reflections**: Are shadows, highlights, and reflections coherent with the light sources?\n"
#                             "- **Texture and surface detail**: Are textures natural and consistent across materials and objects?\n"
#                             "- **Color saturation and uniformity**: Are colors realistic and evenly distributed without artificial patches?\n"
#                             "- **Shape and perspective**: Are object proportions, alignment, and perspective geometrically accurate?\n"
#                             "- **Depth and spatial coherence**: Is there a believable sense of depth and spatial arrangement?\n"
#                             "- **Artifacts and anomalies**: Look for signs of tampering, such as blurred edges, unnatural warping, or inconsistent patterns.\n\n"
#                             "Based on your evaluation of the above criteria, provide a well-reasoned conclusion: Does the image appear to be authentic, or does it show evidence of deepfake or digital manipulation?"
#                         ),
#                     },
#                 ],
#             },
#         ]

#     def forward(self, X):
#         image = X
#         device = self.model.device

#         processed_image = self.image_processor(images=image, return_tensors="pt").to(
#             device
#         )
#         visual_features = self.model.get_image_features(**processed_image)
#         visual_features = torch.mean(torch.stack(visual_features), dim=1)

#         input_ids = self.processor.apply_chat_template(
#             self.get_prompt(Image.fromarray(image)),
#             add_generation_prompt=True,
#             tokenize=True,
#             return_dict=True,
#             return_tensors="pt",
#         ).to(device)

#         output = self.model.generate(
#             **input_ids,
#             do_sample=False,
#             min_new_tokens=self.new_tokens,
#             max_new_tokens=self.new_tokens,
#             use_cache=True,
#             output_hidden_states=True,
#             return_dict_in_generate=True,
#         )

#         # print(self.processor.decode(output.sequences[0]))

#         textual_features = []
#         for layer_index in self.selected_layers:
#             hidden_feature = [
#                 hidden_state[layer_index] for hidden_state in output["hidden_states"]
#             ]
#             hidden_feature = torch.cat(hidden_feature, dim=1)
#             hidden_feature = hidden_feature[:, -self.new_tokens :, :]
#             textual_features.append(hidden_feature)
#         textual_features = torch.cat(textual_features, dim=0)
#         return visual_features, textual_features
