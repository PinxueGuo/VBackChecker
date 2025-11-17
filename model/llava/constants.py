# --------------------------------------------------------
# Visual Instruction Tuning
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee
# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# LISA Questions and GSVA questions

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation mask.",
]

SHORT_QUESTION_LIST_MODE4 = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What are {class_name} in this image? Please respond with segmentation masks.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What are {class_name} in this image? Please output segmentation masks."
]
# SHORT_QUESTION_LIST_MODE4 = [
#     DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment {class_name} in this image? " + "If the image contains an object that exactly matches the description, respond with [SEG]. Otherwise, respond with [REJ] and explain why the object is not present or does not match.",
#     DEFAULT_IMAGE_TOKEN + "\n" + "Please segment {class_name} in this image. " + "If successful, return [SEG]; if not, reply [REJ] and explain why it does not match or is absent.",
#     DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name} in this image? Please respond with segmentation mask. " + "Provide the segmentation mask if found and reply [SEG]. Otherwise, reply [REJ] and detail the reason for the mismatch or absence.",
#     DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name} in this image? Please output segmentation mask. " + "If successful, return [SEG]; if not, reply [REJ] and explain why it does not match or is absent."
# ]
# SHORT_QUESTION_LIST_MODE4 = [
#     DEFAULT_IMAGE_TOKEN + "\n" + "Indicate the object of the following description: {class_name}. Please respond with segmentation mask. " + "If the image contains the object that exactly matches the description, respond with [SEG]. Otherwise, respond with [REJ] and explain why the object is not present or does not match.", 
#     DEFAULT_IMAGE_TOKEN + "\n" + "Segment the object in this image: {class_name}. " + "Respond with [SEG] if the image contains an exact match. If not, respond with [REJ] and explain why no matching object is found.",
#     DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment {class_name}. " + "Respond with [SEG] if there is an exact match. If no matching object is found, reply [REJ] and explain the error reason.",
#     DEFAULT_IMAGE_TOKEN + "\n" + "Please segment {class_name} in this image. " + "If successful, return [SEG]; if not, reply [REJ] and explain why it does not match or is absent.",
#     DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name}? " + "Provide the segmentation mask if found and reply [SEG]. Otherwise, reply [REJ] and detail the reason for the mismatch or absence.",
# ]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explaination.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

ANSWER_LIST_MODE1 = [
    "Here it is.",
    "Sure.",
    "Sure, this is the target.",
    "Sure, here is the segmentation result.",
    "Here you are."
]

ANSWER_LIST_MODE4_START = [
    "The segmentation results are",
    "Sure, they are",
    "Sure,",
    "Sure,",
    "Sure,"
]
ANSWER_LIST_MODE4_TEMPLATE = [
    "{class_name} [SEG]",
    "{class_name}:[SEG]",
    "the mask of {class_name} is [SEG]",
    "the segmentation of {class_name} is [SEG]",
    "the referred {class_name} is [SEG]"
]


# ANSWER_LIST_MODE4_START = [
#     "Sure, ",
#     "The segmentation result of ",
#     "Sure,",
#     "Sure,"
# ]
# ANSWER_LIST_MODE4_TEMPLATE = [
#     "the referred {class_name} is [SEG]",
#     "{class_name} is [SEG]",
#     "the result of {class_name} is [SEG]",
#     "{class_name}: [SEG]"
# ]

ANSWER_LIST_MODE4_END = [
    ".", ".", ".", ".", "."
]


FAITH_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment {class_name}?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment {class_name}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name}? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name}? Please output segmentation mask."
]

FAITH_ANSWER_LIST = [
    "The segmentation result of {class_name} is [SEG].",
    "Sure, {class_name}: [SEG]",
    "Sure, the mask of {class_name} is [SEG].",
    "Sure, the segmentation of {class_name} is [SEG].",
    "Sure, the referred {class_name} is [SEG]."
]
# FAITH_ANSWER_LIST = [
#     "Sure, the referred object is [SEG].",
#     "The segmentation result is [SEG].",
#     "Sure, the result is [SEG].",
#     "Sure, [SEG]",
#     "Sure, the mask is [SEG]."
# ]

FAITH_SIMPLE_ANSWER_LIST = [
    "The segmentation result is [SEG].",
]