import logging
import os
from typing import Union, List
import datasets
import torch
from datasets import load_dataset, concatenate_datasets
import transformers


IGNORE_INDEX = -100  # 定义一个忽略索引值

logger = logging.getLogger('__name__')  # 配置日志记录器

# 定义系统、用户和助手的格式字符串
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
system_format='<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
assistant_format='{content}<|eot_id|>'

def build_instruction_dataset(data_path: Union[List[str],str],
                tokenizer: transformers.PreTrainedTokenizer,
                max_seq_length: int, data_cache_dir = None,
                preprocessing_num_workers = None,
                ):
    """
    构建指令数据集。

    参数:
    - data_path: 包含训练数据文件路径的列表或字符串。
    - tokenizer: 用于分词的transformers.PreTrainedTokenizer。
    - max_seq_length: 最大序列长度。
    - data_cache_dir: 缓存数据集的目录，默认为None，表示使用data_path的目录。
    - preprocessing_num_workers: 预处理时使用的worker数量，默认为None。

    返回:
    - 加工后的数据集。
    """

    def tokenization(examples):
        """
        对例子进行分词处理。

        参数:
        - examples: 包含指令、输入文本和输出的字典。

        返回:
        - 分词后的输入和标签。
        """
        sources = []  # 存储处理后的源序列
        targets = []  # 存储处理后的目标序列
        for instruction, input_text, output in zip(examples['instruction'],examples['input'],examples['output']):
            # 组合指令和输入文本
            if input_text is not None and input_text !="":
                instruction = instruction+'\n'+input_text
            source = system_format.format(content=DEFAULT_SYSTEM_PROMPT) + user_format.format(content=instruction)
            target = output

            sources.append(source)
            targets.append(target)

        # 对源序列和目标序列进行分词
        tokenized_sources = tokenizer(sources, return_attention_mask=False, add_special_tokens=False)
        tokenized_targets = tokenizer(targets, return_attention_mask=False, add_special_tokens=False)

        all_input_ids = []  # 存储所有输入序列的ID
        all_labels = []  # 存储所有标签序列的ID
        for s,t in zip(tokenized_sources['input_ids'],tokenized_targets['input_ids']):
            # 组合源序列和目标序列，并添加标签
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        results = {'input_ids':all_input_ids, 'labels': all_labels}
        return results


    logging.warning("building dataset...")
    all_datasets = []

    if not isinstance(data_path,(list,tuple)):
        data_path = [data_path]  # 转换data_path为列表格式，确保后续处理的一致性

    for file in data_path:  # 遍历数据路径列表

        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))  # 缺省缓存目录设置
        cache_path = os.path.join(data_cache_dir,os.path.basename(file).split('.')[0]+f"_{max_seq_length}")  # 计算缓存文件路径
        os.makedirs(cache_path, exist_ok=True)  # 创建缓存目录

        try:
            processed_dataset = datasets.load_from_disk(cache_path)  # 尝试从磁盘加载处理过的数据集
            logger.info(f'training datasets-{file} has been loaded from disk')
        except Exception:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)  # 加载原始数据集
            tokenization_func = tokenization  # 分词函数
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=["instruction","input","output"],
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            processed_dataset = tokenized_dataset  # 处理数据集
            processed_dataset.save_to_disk(cache_path)  # 保存处理过的数据集到磁盘
        processed_dataset.set_format('torch')  # 设置数据集格式为torch
        all_datasets.append(processed_dataset['train'])  # 将训练集添加到总数据集中

    all_datasets = concatenate_datasets(all_datasets)  # 合并所有训练集
    return all_datasets  # 返回合并后的数据集


'''该函数用于构建指令数据集，以用于训练模型。函数接受以下参数：

data_path：数据文件路径，可以是单个文件或多个文件的列表。
tokenizer：预训练的分词器，用于对数据进行分词。
max_seq_length：最大序列长度，超过该长度的序列将被截断。
data_cache_dir：缓存数据文件的目录，默认为None，表示使用数据文件所在目录。
preprocessing_num_workers：预处理数据时使用的worker数量，默认为None，表示使用默认值。
函数内部定义了一个tokenization函数，用于对每个样本进行分词和标记化处理。处理过程如下：

将指令、输入文本和输出文本合并为一个样本。
使用系统提示符和用户提示符对样本进行格式化。
使用分词器对样本进行分词，并将分词结果转换为指定格式的输入序列和标签序列。
将所有样本的输入序列和标签序列收集到一起，并返回。
函数主要流程如下：

打印警告信息，表示正在构建数据集。
如果data_path不是列表或元组，则将其转换为列表。
遍历每个数据文件，对每个文件进行以下操作：
如果缓存路径不存在，则创建该路径。
尝试从缓存路径加载数据集，如果加载成功，则跳过下面的步骤。
从数据文件加载原始数据集，并使用tokenization函数对数据集进行预处理。
将预处理后的数据集保存到缓存路径。
将预处理后的数据集设置为PyTorch格式。
将所有数据集合并为一个数据集，并返回。
'''